open Machine_ir
module D = Mir_observation.Defect

module Location = struct
  type t = {
    func : Mir_id.Func.t;
    block : Mir_id.Block.t;
    instr : Mir_id.Instr.t option;
  }

  let pp fmt t =
    Fmt.pf fmt "%a %a%a" Mir_id.Func.pp t.func Mir_id.Block.pp t.block
      Fmt.(option (any " " ++ Mir_id.Instr.pp))
      t.instr
end

module Outcome = struct
  type t =
    | Defect of D.t * Location.t
    | Failure of Mir_observation.Row.t
    | Fuel_exhausted
    | Success of Mir_datum.t list
    | Unsupported of string

  let pp fmt = function
    | Defect (d, l) -> Fmt.pf fmt "defect %s at %a" (D.name d) Location.pp l
    | Failure r ->
        Fmt.pf fmt "failure %a(%a)" Mir_failure.pp r.Mir_observation.Row.failure
          Fmt.(list ~sep:(any ", ") Mir_const.pp)
          r.Mir_observation.Row.payload
    | Fuel_exhausted -> Fmt.string fmt "fuel exhausted"
    | Success vs ->
        Fmt.pf fmt "success [%a]" Fmt.(list ~sep:(any ", ") Mir_datum.pp) vs
    | Unsupported s -> Fmt.pf fmt "unsupported %s" s
end

module Binding = struct
  type t = Mir_memory.Key.t Mir_id.Region.Map.t

  let instance t r = Mir_id.Region.Map.find_opt r t

  (* A pointer to a view's first byte, from any stage's program: views are
     stage-independent. *)
  let view_of t (p : (_, _) Mir_program.t) v =
    match Mir_program.find_view p v with
    | None -> None
    | Some view ->
        Option.map
          (fun key ->
            {
              Mir_memory.Pointer.instance = key;
              lo = view.Mir_view.offset;
              hi = Int64.add view.Mir_view.offset view.Mir_view.size;
              perm = view.Mir_view.perm;
              offset = view.Mir_view.offset;
            })
          (instance t view.Mir_view.region)

  let view t (p : Mir_program.generic) v =
    match Mir_program.find_view p v with
    | None -> None
    | Some view ->
        Option.map
          (fun key ->
            {
              Mir_memory.Pointer.instance = key;
              lo = view.Mir_view.offset;
              hi = Int64.add view.Mir_view.offset view.Mir_view.size;
              perm = view.Mir_view.perm;
              offset = view.Mir_view.offset;
            })
          (instance t view.Mir_view.region)
end

let instantiate (p : (_, _) Mir_program.t) memory ~bound =
  List.fold_left
    (fun acc (r : Mir_region.t) ->
      match acc with
      | Error _ -> acc
      | Ok m -> (
          match
            Mir_memory.alloc memory ~region:r.Mir_region.id
              ~size:r.Mir_region.size ~align:r.Mir_region.align ()
          with
          | None ->
              Error
                (Fmt.str "%a: no synthetic address" Mir_id.Region.pp
                   r.Mir_region.id)
          | Some key -> (
              let bytes =
                match r.Mir_region.init with
                | Mir_region.Bound -> bound r.Mir_region.id
                | Mir_region.Constant s -> Some s
                | Mir_region.Uninitialized -> None
              in
              match bytes with
              | Some s
                when Int64.compare
                       (Int64.of_int (String.length s))
                       r.Mir_region.size
                     > 0 ->
                  Error
                    (Fmt.str "%a: too many bytes" Mir_id.Region.pp
                       r.Mir_region.id)
              | Some s ->
                  Mir_memory.write_string memory key ~offset:0L s;
                  Ok (Mir_id.Region.Map.add r.Mir_region.id key m)
              | None -> Ok (Mir_id.Region.Map.add r.Mir_region.id key m))))
    (Ok Mir_id.Region.Map.empty) p.Mir_program.regions

type run = {
  outcome : Outcome.t;
  events : (Mir_event.t * int64) list;
  steps : int64;
  last : Location.t option;
}

(* Why execution stopped early. *)
type stop =
  | Stop_defect of D.t * Location.t
  | Stop_failure of Mir_observation.Row.t
  | Stop_fuel
  | Stop_unsupported of string

type st = {
  program : Mir_program.generic;
  memory : Mir_memory.t;
  binding : Binding.t;
  models : Mir_helper_model.t list;
  invocation : int32 option;
  max_depth : int;
  mutable fuel : int64;
  mutable steps : int64;
  events : int64 array;
  mutable last : Location.t option;
  esc : stop Err.Escape.t;
}

let event_slot = function
  | Mir_event.Emitter -> 0
  | Mir_event.Key -> 1
  | Mir_event.Local -> 2
  | Mir_event.Reduction -> 3
  | Mir_event.Scan -> 4
  | Mir_event.Scan_update -> 5

let stop st s = Err.Escape.throw st.esc s

let tick st loc =
  st.last <- Some loc;
  if Int64.compare st.fuel 0L <= 0 then stop st Stop_fuel;
  st.fuel <- Int64.pred st.fuel;
  st.steps <- Int64.succ st.steps

(* A frame: one function activation's values, by id. *)
type frame = { values : (int, Mir_datum.t) Hashtbl.t; loc : Location.t ref }

let defect st frame d = stop st (Stop_defect (d, !(frame.loc)))

let get st frame (v : Mir_value.t) =
  match Hashtbl.find_opt frame.values (Mir_id.Value.to_int v.Mir_value.id) with
  | Some x -> x
  | None -> defect st frame D.Invalid_program

let set frame (v : Mir_value.t) x =
  Hashtbl.replace frame.values (Mir_id.Value.to_int v.Mir_value.id) x

let bits st frame v =
  match get st frame v with
  | Mir_datum.Bits b -> b
  | Mir_datum.Flags _ | Mir_datum.Order | Mir_datum.Ptr _ ->
      defect st frame D.Invalid_program

let ptr st frame v =
  match get st frame v with
  | Mir_datum.Ptr p -> p
  | Mir_datum.Flags _ | Mir_datum.Order | Mir_datum.Bits _ ->
      defect st frame D.Invalid_program

let width_of st frame (v : Mir_value.t) =
  match v.Mir_value.ty with
  | Mir_type.Int w -> w
  | _ -> defect st frame D.Invalid_program

let float_of st frame (v : Mir_value.t) =
  let b = bits st frame v in
  match v.Mir_value.ty with
  | Mir_type.F64 -> Mir_numeric.f64 b
  | Mir_type.F32 -> Mir_numeric.f32 b
  | _ -> defect st frame D.Invalid_program

(* A float result at the operand precision: binary32 rounds once. *)
let float_result (ty : Mir_type.t) x =
  match ty with
  | Mir_type.F32 -> Mir_numeric.round32 x
  | _ -> Mir_numeric.of_f64 x

let pred b = Mir_datum.Bits (if b then 1L else 0L)

let memory_fault st frame = function
  | Mir_memory.Fault.Bad_access -> defect st frame D.Bad_access
  | Mir_memory.Fault.Uninitialized -> defect st frame D.Uninitialized

let rec call st ~depth (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) args =
  if depth > st.max_depth then stop st Stop_fuel;
  let frame =
    {
      values = Hashtbl.create 64;
      loc =
        ref
          {
            Location.func = f.Mir_func.id;
            block = f.Mir_func.entry;
            instr = None;
          };
    }
  in
  let blocks = Hashtbl.create 16 in
  List.iter
    (fun (b : (_, _) Mir_block.t) ->
      Hashtbl.replace blocks (Mir_id.Block.to_int b.Mir_block.id) b)
    f.Mir_func.blocks;
  let block id = Hashtbl.find blocks (Mir_id.Block.to_int id) in
  let entry = block f.Mir_func.entry in
  (match
     Err.payload
       (Err.List.iter2
          ~unequal_lengths:(fun _ _ -> ())
          (fun p a -> Ok (set frame p a))
          entry.Mir_block.params args)
   with
  | Ok () -> ()
  | Error () -> defect st frame D.Invalid_program);
  set frame entry.Mir_block.order Mir_datum.Order;
  (* every argument is read before any parameter is rebound *)
  let goto (e : Mir_edge.t) =
    let target = block e.Mir_edge.target in
    let values = List.map (get st frame) e.Mir_edge.args in
    List.iter2 (set frame) target.Mir_block.params values;
    set frame target.Mir_block.order Mir_datum.Order;
    target
  in
  let current = ref entry and result = ref None in
  while Option.is_none !result do
    let b = !current in
    List.iter
      (fun (i : Mir_op.t Mir_instr.t) ->
        let loc =
          {
            Location.func = f.Mir_func.id;
            block = b.Mir_block.id;
            instr = Some i.Mir_instr.id;
          }
        in
        frame.loc := loc;
        tick st loc;
        exec st ~depth frame i)
      b.Mir_block.body;
    let loc =
      { Location.func = f.Mir_func.id; block = b.Mir_block.id; instr = None }
    in
    frame.loc := loc;
    tick st loc;
    match b.Mir_block.terminator with
    | Mir_terminator.Jump e -> current := goto e
    | Mir_terminator.Branch { Mir_branch.cond; then_; else_ } ->
        current :=
          goto (if Int64.equal (bits st frame cond) 1L then then_ else else_)
    | Mir_terminator.Return { Mir_return.values; _ } ->
        result := Some (List.map (get st frame) values)
    | Mir_terminator.Fail { Mir_fail.failure; payload; _ } ->
        let payload =
          List.map2
            (fun (v : Mir_value.t) ty ->
              { Mir_const.ty; bits = bits st frame v })
            payload
            (Mir_failure.payload failure)
        in
        stop st
          (Stop_failure
             {
               Mir_observation.Row.failure;
               payload;
               invocation = st.invocation;
               site = None;
             })
  done;
  Option.get !result

and exec st ~depth frame (i : Mir_op.t Mir_instr.t) =
  let put x =
    match i.Mir_instr.results with
    | [ r ] -> set frame r x
    | _ -> defect st frame D.Invalid_program
  in
  (match i.Mir_instr.order with
  | Some o -> set frame o.Mir_order.output Mir_datum.Order
  | None -> ());
  let b = bits st frame and fl = float_of st frame in
  match i.Mir_instr.op with
  | Mir_op.Addr view -> (
      match Binding.view st.binding st.program view with
      | Some p -> put (Mir_datum.Ptr p)
      | None -> defect st frame D.Invalid_program)
  | Mir_op.Bitcast (_, a) -> put (Mir_datum.Bits (b a))
  | Mir_op.Call (callee, args) -> (
      let args = List.map (get st frame) args in
      match callee with
      | Mir_op.Callee.Func id -> (
          match Mir_program.find_func st.program id with
          | Some f ->
              let rs = call st ~depth:(depth + 1) f args in
              List.iter2 (set frame) i.Mir_instr.results rs
          | None -> defect st frame D.Invalid_program)
      | Mir_op.Callee.Helper id -> (
          match Mir_program.find_helper st.program id with
          | None -> defect st frame D.Invalid_program
          | Some h -> (
              match Mir_helper_model.find st.models h with
              | None -> stop st (Stop_unsupported h.Mir_helper.name)
              | Some m -> (
                  match m.Mir_helper_model.run st.memory args with
                  | Mir_helper_model.Returns rs ->
                      if List.length rs <> List.length i.Mir_instr.results then
                        defect st frame D.Invalid_program;
                      List.iter2 (set frame) i.Mir_instr.results rs
                  | Mir_helper_model.Fails (failure, payload) ->
                      if
                        not
                          (List.exists
                             (Mir_failure.equal failure)
                             h.Mir_helper.failures)
                      then defect st frame D.Invalid_program;
                      stop st
                        (Stop_failure
                           {
                             Mir_observation.Row.failure;
                             payload;
                             invocation = st.invocation;
                             site = None;
                           })))))
  | Mir_op.Const c -> put (Mir_datum.of_const c)
  | Mir_op.Copy a -> put (get st frame a)
  | Mir_op.Event (e, n) ->
      let k = event_slot e in
      st.events.(k) <- Int64.add st.events.(k) n
  | Mir_op.Fbinary (o, x, y) ->
      put
        (Mir_datum.Bits
           (float_result x.Mir_value.ty
              (Mir_numeric.float_binary o (fl x) (fl y))))
  | Mir_op.Fcmp (c, x, y) ->
      put (pred (Mir_numeric.float_compare c (fl x) (fl y)))
  | Mir_op.Fconvert (c, a) ->
      put
        (Mir_datum.Bits
           (match c with
           | Mir_op.Fconvert.F32_to_f64 -> Mir_numeric.of_f64 (fl a)
           | Mir_op.Fconvert.F64_to_f32 -> Mir_numeric.round32 (fl a)
           | Mir_op.Fconvert.S64_to_f32 -> Mir_numeric.s64_to_f32 (b a)
           | Mir_op.Fconvert.S64_to_f64 -> Mir_numeric.s64_to_f64 (b a)))
  | Mir_op.Ffma (x, y, z) ->
      put
        (Mir_datum.Bits
           (match x.Mir_value.ty with
           | Mir_type.F32 -> Mir_numeric.fma32 (b x) (b y) (b z)
           | _ -> Mir_numeric.of_f64 (Float.fma (fl x) (fl y) (fl z))))
  | Mir_op.Fto_sint a -> (
      match Mir_numeric.f64_to_s64 (fl a) with
      | Some n -> put (Mir_datum.Bits n)
      | None -> defect st frame D.Domain)
  | Mir_op.Funary (u, a) ->
      put
        (Mir_datum.Bits
           (float_result a.Mir_value.ty (Mir_numeric.float_unary u (fl a))))
  | Mir_op.Iarith (o, x, y) ->
      let w = width_of st frame x in
      let c = b y in
      if
        Mir_op.Iarith.is_shift o
        && (Int64.compare c 0L < 0
           || Int64.compare c (Int64.of_int (Mir_width.bits w)) >= 0)
      then defect st frame D.Domain;
      put (Mir_datum.Bits (Mir_numeric.int_binary o w (b x) c))
  | Mir_op.Icmp (c, x, y) ->
      put (pred (Mir_numeric.int_compare c (width_of st frame x) (b x) (b y)))
  | Mir_op.Idiv (o, x, y) -> (
      match Mir_numeric.int_div o (width_of st frame x) (b x) (b y) with
      | Some r -> put (Mir_datum.Bits r)
      | None -> defect st frame D.Domain)
  | Mir_op.Iext (k, w, a) ->
      let w0 = width_of st frame a in
      put
        (Mir_datum.Bits
           (match k with
           | Mir_op.Iext.Sext ->
               Mir_width.normalize w (Mir_width.signed w0 (b a))
           | Mir_op.Iext.Zext -> b a))
  | Mir_op.Itrunc (w, a) -> put (Mir_datum.Bits (Mir_width.normalize w (b a)))
  | Mir_op.Load { Mir_op.Access.width; addr; align } -> (
      match
        Mir_memory.load st.memory (ptr st frame addr)
          ~bytes:(Mir_width.bytes width) ~align
      with
      | Ok x -> put (Mir_datum.Bits x)
      | Error e -> memory_fault st frame e)
  | Mir_op.Narrow (w, a) ->
      let x = Mir_width.signed (width_of st frame a) (b a) in
      if
        Int64.compare x (Mir_width.min_signed w) < 0
        || Int64.compare x (Mir_width.max_signed w) > 0
      then defect st frame D.Domain
      else put (Mir_datum.Bits (Mir_width.normalize w x))
  | Mir_op.Pbinary (o, x, y) ->
      let x = Int64.equal (b x) 1L and y = Int64.equal (b y) 1L in
      put
        (pred
           (match o with
           | Mir_op.Pbinary.And -> x && y
           | Mir_op.Pbinary.Or -> x || y
           | Mir_op.Pbinary.Xor -> x <> y))
  | Mir_op.Pnot a -> put (pred (not (Int64.equal (b a) 1L)))
  | Mir_op.Ptr_add (p, d) -> (
      match Mir_memory.offset_by (ptr st frame p) (b d) with
      | Some q -> put (Mir_datum.Ptr q)
      | None -> defect st frame D.Bad_access)
  | Mir_op.Select (p, x, y) ->
      put (get st frame (if Int64.equal (b p) 1L then x else y))
  | Mir_op.Store ({ Mir_op.Access.width; addr; align }, v) -> (
      match
        Mir_memory.store st.memory (ptr st frame addr)
          ~bytes:(Mir_width.bytes width) ~align (b v)
      with
      | Ok () -> ()
      | Error e -> memory_fault st frame e)
  | Mir_op.Undef view -> (
      match Binding.view st.binding st.program view with
      | Some p -> Mir_memory.undefine st.memory p
      | None -> defect st frame D.Invalid_program)

let run ?(fuel = 10_000_000L) ?(max_depth = 64) ?invocation ?(models = []) g
    memory binding ~args =
  let program = Mir_verify.Generic.program g in
  let events = Array.make 6 0L in
  let st_ref = ref None in
  let missing =
    List.find_opt
      (fun h -> Option.is_none (Mir_helper_model.find models h))
      program.Mir_program.helpers
  in
  let result =
    match missing with
    | Some h -> Error (Stop_unsupported h.Mir_helper.name)
    | None ->
        Result.map_error Err.Error.kind
          ( Err.Escape.with_escape @@ fun esc ->
            let st =
              {
                program;
                memory;
                binding;
                models;
                invocation;
                max_depth;
                fuel;
                steps = 0L;
                events;
                last = None;
                esc;
              }
            in
            st_ref := Some st;
            match Mir_program.find_func program program.Mir_program.main with
            | Some f -> call st ~depth:0 f args
            | None -> invalid_arg "Mir_interp: a verified program has its main"
          )
  in
  let outcome =
    match result with
    | Ok vs -> Outcome.Success vs
    | Error (Stop_defect (d, l)) -> Outcome.Defect (d, l)
    | Error (Stop_failure r) -> Outcome.Failure r
    | Error Stop_fuel -> Outcome.Fuel_exhausted
    | Error (Stop_unsupported s) -> Outcome.Unsupported s
  in
  let steps, last =
    match !st_ref with Some st -> (st.steps, st.last) | None -> (0L, None)
  in
  {
    outcome;
    events = List.map (fun e -> (e, events.(event_slot e))) Mir_event.all;
    steps;
    last;
  }

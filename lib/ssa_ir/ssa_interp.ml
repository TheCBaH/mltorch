module Counters = struct
  type t = { mutable loads : int; mutable stores : int; marks : int array }

  let create () = { loads = 0; stores = 0; marks = Array.make 6 0 }

  let slot = function
    | Ssa_mark.Emitter -> 0
    | Ssa_mark.Key -> 1
    | Ssa_mark.Local -> 2
    | Ssa_mark.Reduction -> 3
    | Ssa_mark.Scan -> 4
    | Ssa_mark.Scan_update -> 5

  let mark t m = t.marks.(slot m)
  let loads t = t.loads
  let stores t = t.stores
  let count t m = t.marks.(slot m) <- t.marks.(slot m) + 1
end

type failure =
  [ `Coord_out_of_range of Expr.Source.t * Expr.Axis.t * int * int Expr.Coord.t
  | `Index_overflow of Expr.Index_overflow.t ]

type error = [ failure | `Invalid_program of Ssa_verify.diagnostic ]

let pp_error fmt : [< error ] -> unit = function
  | `Coord_out_of_range (src, axis, v, c) ->
      Fmt.pf fmt "%a: coordinate %a = %d is outside the buffer, at (%a)"
        Expr.Source.pp src Expr.Axis.pp axis v (Expr.Coord.pp Fmt.int) c
  | `Index_overflow { Expr.Index_overflow.op; lhs; rhs } ->
      Fmt.pf fmt "index %s overflows on %d and %d"
        (match op with `Add -> "add" | `Mul -> "mul" | `Sub -> "sub")
        lhs rhs
  | `Invalid_program d -> Ssa_verify.pp_diagnostic fmt d

(* A runtime value. An index and an i64 are both [I], a binary32 and a binary64
   both [F]: the verifier has already fixed which a slot holds, and a binary32
   value is kept rounded. *)
type v = E | F of float | I of int64 | P of bool

type st = {
  esc : failure Err.Escape.t;
  program : Ssa_program.t;
  memory : Ssa_memory.t;
  counters : Counters.t;
  env : v array;
}

let get st (x : Ssa_value.t) = st.env.((x.Ssa_value.id :> int))
let set st (x : Ssa_value.t) v = st.env.((x.Ssa_value.id :> int)) <- v

let float_of st x =
  match get st x with
  | F f -> f
  | E | I _ | P _ -> invalid_arg "Ssa_interp: float"

let int_of st x =
  match get st x with
  | I i -> i
  | E | F _ | P _ -> invalid_arg "Ssa_interp: int"

let bool_of st x =
  match get st x with
  | P b -> b
  | E | F _ | I _ -> invalid_arg "Ssa_interp: pred"

let is_f32 (x : Ssa_value.t) =
  Ssa_type.equal x.Ssa_value.ty (Ssa_type.Scalar Ssa_type.F32)

let const : Ssa_const.t -> v = function
  | Ssa_const.F32 x | Ssa_const.F64 x -> F x
  | Ssa_const.I64 x | Ssa_const.Index x -> I x
  | Ssa_const.Pred b -> P b

let buffer st id =
  match Ssa_program.find_buffer st.program id with
  | Some b -> b
  | None -> invalid_arg "Ssa_interp: undeclared buffer"

let cells st id =
  match Ssa_memory.find st.memory id with
  | Some c -> c
  | None -> invalid_arg "Ssa_interp: buffer is not in memory"

let coord_of st c = Expr.Coord.map (int_of st) c

(* The element offset an access names, after the checks the access owes. A
   coordinate load reports the first axis outside; a flat offset or any store
   outside is a defect, not a row. *)
let element st (b : Ssa_buffer.t) (at : Ssa_access.t) ~checked =
  match at with
  | Ssa_access.Coord c -> (
      let c = coord_of st c in
      match Ssa_memory.first_outside b.Ssa_buffer.extents c with
      | None -> Int64.to_int (Ssa_memory.offset b.Ssa_buffer.extents c)
      | Some axis ->
          if checked then
            Err.Escape.throw st.esc
              (`Coord_out_of_range
                 ( Ssa_buffer.source b,
                   axis,
                   Int64.to_int (Expr.Coord.get c axis),
                   Expr.Coord.map Int64.to_int c )
                : failure)
          else invalid_arg "Ssa_interp: store outside its buffer")
  | Ssa_access.Flat o -> (
      let o = int_of st o in
      match Ssa_buffer.elements b.Ssa_buffer.extents with
      | Some n when Int64.compare o 0L >= 0 && Int64.compare o n < 0 ->
          Int64.to_int o
      | Some _ | None ->
          invalid_arg "Ssa_interp: flat access outside its buffer")

let overflow st op lhs rhs =
  Err.Escape.throw st.esc
    (`Index_overflow
       {
         Expr.Index_overflow.op;
         lhs = Int64.to_int lhs;
         rhs = Int64.to_int rhs;
       }
      : failure)

let checked_index st op lhs rhs r =
  if Ssa_const.in_index_domain r then I r else overflow st op lhs rhs

let exec_op st (i : Ssa_instr.t) =
  let result =
    match i.Ssa_instr.results with
    | r :: _ -> r
    | [] -> invalid_arg "Ssa_interp: no result"
  in
  let put v = set st result v in
  match i.Ssa_instr.op with
  | Ssa_op.Const c -> put (const c)
  | Ssa_op.Convert (c, a) -> (
      match c with
      | Ssa_op.Convert.F32_to_f64 -> put (F (float_of st a))
      | Ssa_op.Convert.F64_to_f32 ->
          put (F (Ssa_const.round_f32 (float_of st a)))
      | Ssa_op.Convert.Index_to_f64 -> put (F (Int64.to_float (int_of st a)))
      | Ssa_op.Convert.Index_to_i64 -> put (I (int_of st a)))
  | Ssa_op.Float_binary (op, a, b) ->
      let x = float_of st a in
      let y = float_of st b in
      let r = Expr.Value.apply_binary op x y in
      put (F (if is_f32 result then Ssa_const.round_f32 r else r))
  | Ssa_op.Index_add (a, b) ->
      let x = int_of st a in
      let y = int_of st b in
      put (checked_index st `Add x y (Int64.add x y))
  | Ssa_op.Index_scale (k, a) ->
      let y = int_of st a in
      put (checked_index st `Mul k y (Int64.mul k y))
  | Ssa_op.Load { buffer = id; at; decode } -> (
      let b = buffer st id in
      let at = element st b at ~checked:true in
      st.counters.Counters.loads <- st.counters.Counters.loads + 1;
      match (cells st id, decode) with
      | ( Ssa_memory.Floats a,
          (Ssa_op.Decode.Bool_to_f64 | Ssa_op.Decode.F32_to_f64) ) ->
          put (F a.(at))
      | Ssa_memory.Int64s a, Ssa_op.Decode.I64 -> put (I a.(at))
      | (Ssa_memory.Floats _ | Ssa_memory.Int64s _), _ ->
          invalid_arg "Ssa_interp: decode does not match the cells")
  | Ssa_op.Mark m -> Counters.count st.counters m
  | Ssa_op.Store { buffer = id; at; encode; value } -> (
      let b = buffer st id in
      let at = element st b at ~checked:false in
      st.counters.Counters.stores <- st.counters.Counters.stores + 1;
      match (cells st id, encode) with
      | Ssa_memory.Floats a, Ssa_op.Encode.Bool_nonzero ->
          a.(at) <- (if float_of st value <> 0. then 1. else 0.)
      | Ssa_memory.Floats a, Ssa_op.Encode.F32_round ->
          a.(at) <- Ssa_const.round_f32 (float_of st value)
      | Ssa_memory.Int64s a, Ssa_op.Encode.I64 -> a.(at) <- int_of st value
      | (Ssa_memory.Floats _ | Ssa_memory.Int64s _), _ ->
          invalid_arg "Ssa_interp: encode does not match the cells")

(* Run a region's statements; the caller reads the yields. *)
let rec region st (r : Ssa_region.t) = List.iter (stmt st) r.Ssa_region.body
and yields st (r : Ssa_region.t) = List.map (get st) r.Ssa_region.yields

and stmt st : Ssa_region.t Ssa_stmt.t -> unit = function
  | Ssa_stmt.For { lo; hi; step; inits; results; body } ->
      let lo = int_of st lo in
      let hi = int_of st hi in
      let iv, carried =
        match body.Ssa_region.params with
        | iv :: carried -> (iv, carried)
        | [] -> invalid_arg "Ssa_interp: loop without an induction value"
      in
      let state = ref (List.map (get st) inits) in
      let i = ref lo in
      while Int64.compare !i hi < 0 do
        set st iv (I !i);
        List.iter2 (set st) carried !state;
        region st body;
        (* every yield is read before any parameter is rebound *)
        state := yields st body;
        i := Int64.add !i step
      done;
      List.iter2 (set st) results !state
  | Ssa_stmt.If { cond; results; then_; else_ } ->
      let r = if bool_of st cond then then_ else else_ in
      region st r;
      List.iter2 (set st) results (yields st r)
  | Ssa_stmt.Instr i -> exec_op st i
  | Ssa_stmt.Ordered_sum { lo; hi; seed; token = _; results; body } ->
      let lo = int_of st lo in
      let hi = int_of st hi in
      let iv =
        match body.Ssa_region.params with
        | iv :: _ -> iv
        | [] -> invalid_arg "Ssa_interp: sum without an induction value"
      in
      let narrow = if is_f32 seed then Ssa_const.round_f32 else Fun.id in
      let acc = ref (float_of st seed) in
      let i = ref lo in
      while Int64.compare !i hi < 0 do
        set st iv (I !i);
        List.iter
          (fun p ->
            if Ssa_type.equal p.Ssa_value.ty Ssa_type.Effect then set st p E)
          body.Ssa_region.params;
        region st body;
        (match yields st body with
        | F term :: _ -> acc := narrow (!acc +. term)
        | _ -> invalid_arg "Ssa_interp: a sum term is not a float");
        i := Int64.add !i 1L
      done;
      List.iter2 (set st) results [ F !acc; E ]

let run ?(counters = Counters.create ()) (p : Ssa_program.t) ~memory =
  match Err.payload (Ssa_verify.check p) with
  | Error (`Invalid_program d) -> Err.fail (`Invalid_program d : error)
  | Ok () ->
      Err.map_error
        (fun (f : failure) -> (f :> error))
        ( Err.Escape.with_escape @@ fun esc ->
          let st =
            {
              esc;
              program = p;
              memory;
              counters;
              env = Array.make (p.Ssa_program.next_value :> int) E;
            }
          in
          region st p.Ssa_program.entry )

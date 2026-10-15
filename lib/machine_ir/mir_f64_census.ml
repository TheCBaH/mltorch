module Role = struct
  type t = Arithmetic | Conversion | Helper_call | Memory_move

  let all = [ Arithmetic; Conversion; Helper_call; Memory_move ]

  let name = function
    | Arithmetic -> "arithmetic"
    | Conversion -> "conversion"
    | Helper_call -> "helper_call"
    | Memory_move -> "memory_move"
end

module String_map = Map.Make (String)

type t = {
  arithmetic : int;
  conversion : int;
  helper_calls : int String_map.t;
  memory_move : int;
}

let empty =
  {
    arithmetic = 0;
    conversion = 0;
    helper_calls = String_map.empty;
    memory_move = 0;
  }

let is_f64 (ty : Mir_type.t) =
  match ty with
  | Mir_type.F64 | Mir_type.Vec (Mir_type.Elem.F64, _) -> true
  | Mir_type.F32 | Mir_type.Flags | Mir_type.Int _ | Mir_type.Mask _
  | Mir_type.Order | Mir_type.Pred | Mir_type.Ptr
  | Mir_type.Vec ((Mir_type.Elem.F32 | Mir_type.Elem.Int _), _) ->
      false

let touches (i : Mir_op.t Mir_instr.t) =
  let value (v : Mir_value.t) = is_f64 v.Mir_value.ty in
  List.exists value i.Mir_instr.results
  || List.exists value (Mir_op.operands i.Mir_instr.op)

let callee_name (p : Mir_program.generic) = function
  | Mir_op.Callee.Func f -> (
      match Mir_program.find_func p f with
      | Some fn -> fn.Mir_func.name
      | None -> Fmt.str "%a" Mir_id.Func.pp f)
  | Mir_op.Callee.Helper h -> (
      match Mir_program.find_helper p h with
      | Some hp -> hp.Mir_helper.name
      | None -> Fmt.str "%a" Mir_id.Helper.pp h)

let bump_call name t =
  let n = Option.value ~default:0 (String_map.find_opt name t.helper_calls) in
  { t with helper_calls = String_map.add name (n + 1) t.helper_calls }

let add p t (i : Mir_op.t Mir_instr.t) =
  if not (touches i) then t
  else
    match i.Mir_instr.op with
    | Mir_op.Fbinary _ | Mir_op.Fcmp _ | Mir_op.Ffma _ | Mir_op.Funary _ ->
        { t with arithmetic = t.arithmetic + 1 }
    | Mir_op.Fconvert _ | Mir_op.Fto_sint _ ->
        { t with conversion = t.conversion + 1 }
    | Mir_op.Call (c, _) -> bump_call (callee_name p c) t
    | Mir_op.Bitcast _ | Mir_op.Const _ | Mir_op.Copy _ | Mir_op.Load _
    | Mir_op.Select _ | Mir_op.Store _ | Mir_op.Vconcat _ | Mir_op.Vextract _
    | Mir_op.Vinsert _ | Mir_op.Vload _ | Mir_op.Vslice _ | Mir_op.Vsplat _
    | Mir_op.Vstore _ ->
        { t with memory_move = t.memory_move + 1 }
    | Mir_op.Addr _ | Mir_op.Event _ | Mir_op.Iarith _ | Mir_op.Icmp _
    | Mir_op.Idiv _ | Mir_op.Iext _ | Mir_op.Itrunc _ | Mir_op.Narrow _
    | Mir_op.Pbinary _ | Mir_op.Pnot _ | Mir_op.Ptr_add _ | Mir_op.Undef _ ->
        (* no [f64] operand or result is typeable here *)
        t

let of_program g =
  let p = Mir_verify.Generic.program g in
  List.fold_left
    (fun t (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) ->
      List.fold_left
        (fun t (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
          List.fold_left (add p) t b.Mir_block.body)
        t f.Mir_func.blocks)
    empty p.Mir_program.funcs

let helpers t = String_map.bindings t.helper_calls

let count t = function
  | Role.Arithmetic -> t.arithmetic
  | Role.Conversion -> t.conversion
  | Role.Helper_call ->
      String_map.fold (fun _ n acc -> acc + n) t.helper_calls 0
  | Role.Memory_move -> t.memory_move

let total t = List.fold_left (fun acc r -> acc + count t r) 0 Role.all

let pp ppf t =
  Fmt.pf ppf "arithmetic %d, conversion %d, helper_call %d%a, memory_move %d"
    t.arithmetic t.conversion (count t Role.Helper_call)
    (fun ppf -> function
      | [] -> ()
      | hs ->
          Fmt.pf ppf " (%a)"
            Fmt.(list ~sep:comma (pair ~sep:(any " x") string int))
            hs)
    (helpers t) t.memory_move

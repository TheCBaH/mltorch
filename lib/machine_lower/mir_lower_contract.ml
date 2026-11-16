(* The contraction helper calls: a contraction on whole buffers, and a view
   call as calls on offset addresses. *)

open Ssa_ir
open Machine_ir
open Mir_lower_core
module B = Mir_builder
module L = Mir_layout_map

(* A call to the contraction helper of [desc] on three operand addresses,
   declaring the helper at its first call. *)
let contract_call st (desc : Contraction.t) pa pb pc =
  let same d = String.equal (Contraction.name d) (Contraction.name desc) in
  let rec find k = function
    | [] -> None
    | d :: rest -> if same d then Some k else find (k + 1) rest
  in
  let index =
    match find 0 st.contracts with
    | Some k -> k
    | None ->
        st.contracts <- st.contracts @ [ desc ];
        List.length st.contracts - 1
  in
  let h =
    Mir_contract.descriptor ~index:(Mir_contract.Index.of_int index) desc
  in
  let signature = function
    | Mir_op.Callee.Helper _ ->
        Some { Mir_typing.Signature.params = h.Mir_helper.params; results = [] }
    | Mir_op.Callee.Func _ -> None
  in
  match
    B.op ~origin:st.origin st.bld st.cur ~signature
      (Mir_op.Call (Mir_op.Callee.Helper h.Mir_helper.id, [ pa; pb; pc ]))
  with
  | Ok [] -> ()
  | _ -> invalid_arg "Mir_lower: a contraction helper call"

(* A view call as calls of the contraction helpers on offset addresses: each
   operand's address is its buffer's plus its base and a static element offset,
   in element bytes. One base per operand is one call of the descriptor's own
   helper; per-row bases split the call along the axis they name, one call per
   row (A's and C's [m], B's [n]) of the descriptor made single there, since a
   helper takes one address per operand. *)
let view_call st (call : Ssa_view_call.t) =
  let d = call.Ssa_view_call.desc in
  let rows (v : Ssa_view_call.View.t) =
    match v.Ssa_view_call.View.at with
    | Ssa_view_call.At.Rows _ -> true
    | Ssa_view_call.At.Base _ -> false
  in
  let split_m = rows call.Ssa_view_call.a || rows call.Ssa_view_call.c in
  let split_n = rows call.Ssa_view_call.b in
  let sub =
    Result.get_ok
      (Err.payload
         (Contraction.make ~batch:d.Contraction.batch
            ~m:(if split_m then 1L else d.Contraction.m)
            ~n:(if split_n then 1L else d.Contraction.n)
            ~k:d.Contraction.k ~a:d.Contraction.a ~b:d.Contraction.b
            ~c:d.Contraction.c ~format:d.Contraction.format
            ~seed:d.Contraction.seed ~numerics:d.Contraction.numerics))
  in
  (* The address of the cells call [row] of the split axis starts at: a
     single base moves by [stride] per row (zero rows where the axis is not
     split), a per-row base replaces it; [extra] elements more. *)
  let address (v : Ssa_view_call.View.t) ~row ~stride ~extra =
    with_role st Mir_origin.Role.Address;
    let e = entry_of st v.Ssa_view_call.View.buffer in
    let base, moved =
      match v.Ssa_view_call.View.at with
      | Ssa_view_call.At.Base b ->
          ( b,
            if mutated st Mutation.View_outer_dropped then 0L
            else Int64.mul (Int64.of_int row) stride )
      | Ssa_view_call.At.Rows rs -> (List.nth rs row, 0L)
    in
    let offset = Int64.add moved extra in
    let element = sext st (value st base) in
    let element =
      if Int64.equal offset 0L then element
      else emit st (Mir_op.Iarith (Mir_op.Iarith.Add, element, i64k st offset))
    in
    let bytes =
      emit st
        (Mir_op.Iarith (Mir_op.Iarith.Mul, element, i64k st e.L.Entry.elem_bytes))
    in
    let start = emit st (Mir_op.Addr e.L.Entry.view) in
    if mutated st Mutation.View_base_dropped then start
    else emit st (Mir_op.Ptr_add (start, bytes))
  in
  let count split n = if split then Int64.to_int n else 1 in
  let col = d.Contraction.c.Contraction.Output.col in
  for i = 0 to count split_m d.Contraction.m - 1 do
    for j = 0 to count split_n d.Contraction.n - 1 do
      let pa =
        address call.Ssa_view_call.a ~row:i
          ~stride:d.Contraction.a.Contraction.Operand.outer ~extra:0L
      in
      let pb =
        address call.Ssa_view_call.b ~row:j
          ~stride:d.Contraction.b.Contraction.Operand.outer ~extra:0L
      in
      let pc =
        address call.Ssa_view_call.c ~row:i
          ~stride:d.Contraction.c.Contraction.Output.row
          ~extra:(Int64.mul (Int64.of_int j) col)
      in
      contract_call st sub pa pb pc
    done
  done

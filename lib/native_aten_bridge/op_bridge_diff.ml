open Pytorch_types
open Op_bridge_error
open Op_bridge_decode

let dispatch ~aten_env (node : Node.t) =
  if node.target <> "torch.ops.aten.diff.default" then None
  else
    Some
      (let* x = tensor_arg aten_env node "self" in
       let rank = aten_rank x in
       let* dim = dim_arg ~default:(-1) node "dim" in
       let* axis = dim_axis ~op:"diff.default" ~rank dim in
       let* order = int_arg ~default:1 node "n" in
       let optional arg_name =
         let* t = decode_result (D.tensor_arg_opt aten_env node arg_name) in
         if Ctypes.is_null t then return None
         else
           let got = aten_rank t in
           if not (Rank.equal rank got) then
             fail
               (`Operand_rank { Operand_rank.arg_name; expected = rank; got })
           else
             let* value = native_of_aten arg_name t in
             return (Some value)
       in
       let* prepend = optional "prepend" in
       let* append = optional "append" in
       let* x = native_of_aten "self" x in
       let params = Diff.Diff.{ axis; order = Aten_int.Size.of_int order } in
       let inputs = Diff.Diff.segments ~x ~prepend ~append in
       build_g ~name:"diff" inputs (fun ids ->
           let open Graph_builder in
           let x = List.nth ids (if Option.is_some prepend then 1 else 0) in
           let prepend =
             if Option.is_some prepend then Some (List.hd ids) else None
           in
           let append =
             if Option.is_some append then Some (List.hd (List.rev ids))
             else None
           in
           let+ y = diff params x ?prepend ?append () in
           [ y ]))

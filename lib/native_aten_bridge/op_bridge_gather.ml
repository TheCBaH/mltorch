(* Gather, scatter and index family (select/select_scatter/index/embedding/repeat_interleave) op-dispatch arms for [Op_bridge], split from op_bridge_shape.ml. See
   op_bridge.ml for the [dispatch] facade that folds every family module
   (including this one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode
module D = Interp_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten.select.int" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_x in
         let rank = aten_rank aten_x in
         let* dim = dim_arg node "dim" in
         let* index = int_arg node "index" in
         let index = Aten_int.Index.of_int index in
         let* x = native_of_aten "self" aten_x in
         let* d = norm_dim ~op:"select.int" ~rank dim in
         let axis = Aten_shape.axis_of_dim ~rank (Aten_int.Dim.of_int d) in
         let extent = Vec6.get (packed_shape x) axis in
         let* idx =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_index ~extent ~index)
         in
         build_g ~name:"select" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = select { Split.Select.axis; index = idx } x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* One [Select_scatter] node: [self] with [src] written at [idx] along
     [axis], and every other position carried through from [self] unchanged
     -- the write-back counterpart of [select.int] just above, sharing its
     [Aten_shape.resolve_index] canonicalisation (ATen REJECTS an
     out-of-range index rather than clamping it). [index] is declared
     [SymInt] in ATen's own schema, same as [select.int]'s plain [int] once
     traced: the corpus's own occurrences all arrive as [as_int], so
     [int_arg] reads it unchanged. *)
  | "torch.ops.aten.select_scatter.default" ->
      Some
        (let* aten_self = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_self in
         let* aten_src = tensor_arg aten_env node "src" in
         let* () = require_f32 "src" aten_src in
         let rank = aten_rank aten_self in
         let* dim = dim_arg node "dim" in
         let* index = int_arg node "index" in
         let index = Aten_int.Index.of_int index in
         let* self = native_of_aten "self" aten_self in
         let* src = native_of_aten "src" aten_src in
         let* d = norm_dim ~op:"select_scatter.default" ~rank dim in
         let axis = Aten_shape.axis_of_dim ~rank (Aten_int.Dim.of_int d) in
         let extent = Vec6.get (packed_shape self) axis in
         let* idx =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_index ~extent ~index)
         in
         build_g ~name:"select_scatter" [ self; src ] (function
           | [ self_id; src_id ] ->
               let open Graph_builder in
               let+ y =
                 select_scatter
                   { Split.Select_scatter.axis; index = idx }
                   ~self:self_id ~src:src_id
               in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* No [require_f32]: legalizes to [Reshape] alone (see [squeeze.dim]'s own
     comment below), and [Reshape]'s [Eval_direct] arm now dispatches an I64
     operand through its own exact [Compute_i64] path -- and, since this
     session, [Graph_builder.reshape] itself declares the OUTPUT edge's
     format/quant from the operand's, not [op1]'s F32 default, so a
     downstream dtype-branching consumer sees the correct signature too. Any
     dtype [native_of_aten] cannot represent (i.e. not F32/F64/I64) is
     already rejected by its own [Unsupported_dtype], with no separate gate
     needed here -- matching [view.default]/[_unsafe_view.default], which
     never had one. *)
  | "torch.ops.aten.repeat_interleave.self_int" ->
      Some
        (let op = "repeat_interleave.self_int" in
         let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dim = int_opt_arg node "dim" in
         let* dim =
           match dim with
           | Some d -> return d
           | None ->
               fail
                 (`Validation_failure
                    (op ^ ": dim=None (flatten-first) is not yet supported"))
         in
         let* axis = dim_axis ~op ~rank (Aten_int.Dim.of_int dim) in
         let* repeats_int = int_arg node "repeats" in
         let* repeats = pos ~op ~param:`Repeats repeats_int in
         let* x = native_of_aten "self" t in
         build_g ~name:"repeat_interleave" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y =
                 repeat_interleave
                   { Repeat.RepeatInterleave.axis; repeats }
                   x_id
               in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* `_assert_tensor_metadata(Tensor a, SymInt[]? size=None, SymInt[]?
     stride=None, ScalarType? dtype=None, *, Device? device=None, Layout?
     layout=None) -> ()`: a pure debug assertion the exporter inserts to pin
     a tensor's metadata -- it returns no [Tensor] at all (this repo's own
     PT2 producer always serializes it with an empty node [outputs] list),
     so [build_g]'s own body returns [[]] rather than [[y]]. Given an
     already-successfully-exported graph with shapes resolved at parse time
     the assertion is unconditionally true, so [a] is routed to the same
     [Discard] sink [max_pool2d_with_indices.default]'s dead indices edge
     uses rather than left dangling. [size]/[stride]/[dtype]/[device]/
     [layout] restate facts already true of [a] and have no existing typed
     decoder (unlike [implicit]'s or [cudnn_enabled]'s plain [bool_arg]) --
     decoding them would add new accessor machinery for values Native never
     acts on, so they are left unread, matching [Native_interp]'s arm. No
     real ATen C call is made here: unlike every other bridge arm, this op's
     "computation" is discarding a tensor [aten_env] already holds from an
     earlier node's real dispatch, so there is nothing to link against and
     no [bin/aten_ops_gen.ml] entry is needed. *)
  | "torch.ops.aten.embedding.default" ->
      Some
        (let* weight_t = tensor_arg aten_env node "weight" in
         let* indices_t = tensor_arg aten_env node "indices" in
         if aten_rank weight_t <> Rank.of_int 2 then
           fail (`Validation_failure "embedding: weight must be 2-D")
         else if Aten_tensor.scalar_type indices_t <> Aten_scalar_type.Long then
           fail (`Validation_failure "embedding: indices must be int64")
         else
           let indices_rank = aten_rank indices_t in
           let* padding_idx =
             decode_result (D.int_arg_result ~default:(-1) node "padding_idx")
           in
           let* weight_n = native_of_aten "weight" weight_t in
           let* indices_n = native_of_aten "indices" indices_t in
           build_g ~name:"embedding" [ weight_n; indices_n ] (function
             | [ weight_id; indices_id ] ->
                 let open Graph_builder in
                 let+ y =
                   embedding
                     {
                       Embedding.Embedding.indices_rank;
                       padding_idx = Aten_int.Index.of_int padding_idx;
                     }
                     ~weight:weight_id ~indices:indices_id
                 in
                 [ y ]
             | operands -> arity_mismatch ~op:node.Node.target operands))
  | "torch.ops.aten.index.Tensor" ->
      Some
        (let* self_t = tensor_arg aten_env node "self" in
         let rank = aten_rank self_t in
         let* entries =
           decode_result (D.optional_tensors_arg_result aten_env node "indices")
         in
         let got = List.length entries in
         let index_list_fail (fault : Op_bridge_error.Index_list.fault) =
           fail (`Index_list { Op_bridge_error.Index_list.fault })
         in
         if got > (rank :> int) then
           index_list_fail
             (Op_bridge_error.Index_list.Length_mismatch
                { expected = rank; got })
         else
           let live_positions =
             List.mapi (fun i e -> (i, e)) entries
             |> List.filter_map (fun (i, e) ->
                 if Option.is_some e then Some i else None)
           in
           match live_positions with
           | [] -> index_list_fail Op_bridge_error.Index_list.No_live_entry
           | _ :: _ :: _ ->
               index_list_fail
                 (Op_bridge_error.Index_list.Multiple_live_entries
                    live_positions)
           | [ p ] ->
               let index_t = Option.get (List.nth entries p) in
               let dtype = Aten_tensor.scalar_type index_t in
               if not (dtype = Aten_scalar_type.Long) then
                 index_list_fail
                   (Op_bridge_error.Index_list.Wrong_dtype
                      { position = p; dtype })
               else
                 let index_rank = aten_rank index_t in
                 if (index_rank :> int) < 1 then
                   index_list_fail
                     (Op_bridge_error.Index_list.Wrong_rank
                        { position = p; rank = index_rank })
                 else
                   let* axis =
                     dim_axis ~op:"index.Tensor" ~rank (Aten_int.Dim.of_int p)
                   in
                   let* self_n = native_of_aten "self" self_t in
                   let* index_n = native_of_aten "index" index_t in
                   build_g ~name:"index_tensor" [ self_n; index_n ] (function
                     | [ self_id; index_id ] ->
                         let open Graph_builder in
                         let+ y =
                           index_tensor
                             { Index_tensor.Index_tensor.axis; index_rank }
                             ~self:self_id ~index:index_id
                         in
                         [ y ]
                     | operands -> arity_mismatch ~op:node.Node.target operands))
  | _ -> None

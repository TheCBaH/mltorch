(* Scaled dot-product attention op-dispatch arms for [Op_bridge], split from op_bridge.ml. See op_bridge.ml for the [dispatch] facade that
   folds every family module (including this one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  (* op8-impl.md commit 3. Both arms (this one and Native_interp's) implement
     the SAME contract and neither calls ATen: [dropout_p]/[is_causal]/
     [enable_gqa] are typed rejections rather than fields [Attention.Sdpa.t]
     carries (op8-impl.md's [params] has only [scale]); a boolean mask is
     rejected rather than reinterpreted as f32 (F10); Q/K/V/mask rank is
     checked on the RAW ATen tensor, before [native_of_aten] erases it
     (F13) -- the only place it can be, since [Tensor_sig.t] keeps no rank
     field for [Attention.Sdpa.output_shape] to check downstream.
     [value.C <> query.C] (Ev <> E) and an inadmissible mask broadcast shape
     are NOT re-checked here: both are flash-oracle boundaries (F4), not
     rank facts, and [Attention.Sdpa.output_shape] already rejects them (via
     [Graph_builder.build]'s [`Build]) with the same [Shape_error.Sdpa] rows
     [Native_interp] reaches through the same op. *)
  | "torch.ops.aten.scaled_dot_product_attention.default" ->
      Some
        (let* query = tensor_arg aten_env node "query" in
         let* key = tensor_arg aten_env node "key" in
         let* value = tensor_arg aten_env node "value" in
         let* () = require_rank "sdpa query" ~expected:4 query in
         let* () = require_rank "sdpa key" ~expected:4 key in
         let* () = require_rank "sdpa value" ~expected:4 value in
         let* () = require_f32 "sdpa query" query in
         let* () = require_f32 "sdpa key" key in
         let* () = require_f32 "sdpa value" value in
         let* dropout_p = float_arg ~default:0.0 node "dropout_p" in
         let* () =
           if Float.equal dropout_p 0.0 then return ()
           else fail (`Sdpa_reject (Attention.Sdpa.Reject.Dropout dropout_p))
         in
         let* is_causal = bool_arg ~default:false node "is_causal" in
         let* () =
           if is_causal then fail (`Sdpa_reject Attention.Sdpa.Reject.Causal)
           else return ()
         in
         let* enable_gqa = bool_arg ~default:false node "enable_gqa" in
         let* () =
           if enable_gqa then fail (`Sdpa_reject Attention.Sdpa.Reject.Gqa)
           else return ()
         in
         let* scale_opt = float_opt_arg node "scale" in
         let* scale =
           match scale_opt with
           | None -> return Attention.Sdpa.Scale.Default
           | Some s ->
               let* () =
                 if Float.is_finite s then return ()
                 else
                   fail
                     (`Sdpa_reject (Attention.Sdpa.Reject.Non_finite_scale s))
               in
               let* () =
                 if Float.compare s 0.0 < 0 then
                   fail (`Sdpa_reject (Attention.Sdpa.Reject.Negative_scale s))
                 else return ()
               in
               return (Attention.Sdpa.Scale.Explicit s)
         in
         let* mask_opt =
           if optional_tensor_present node "attn_mask" then
             let* m = tensor_arg aten_env node "attn_mask" in
             let* () =
               match Aten_tensor.scalar_type m with
               | Aten_scalar_type.Bool ->
                   fail (`Sdpa_reject Attention.Sdpa.Reject.Boolean_mask)
               | _ -> return ()
             in
             let* () = require_f32 "sdpa attn_mask" m in
             let got = aten_rank m in
             let* () =
               if got = 2 || got = 4 then return ()
               else
                 fail
                   (`Sdpa_reject
                      (Attention.Sdpa.Reject.Rank
                         {
                           arg_name = "sdpa attn_mask";
                           expected = [ 2; 4 ];
                           got;
                         }))
             in
             let* mask = native_of_aten "attn_mask" m in
             return (Some mask)
           else return None
         in
         let* q = native_of_aten "query" query in
         let* k = native_of_aten "key" key in
         let* v = native_of_aten "value" value in
         let params = { Attention.Sdpa.scale } in
         build_g ~name:"sdpa"
           ([ q; k; v ] @ Option.to_list mask_opt)
           (function
             | [ q_id; k_id; v_id ] ->
                 let open Graph_builder in
                 let+ y = sdpa params ~query:q_id ~key:k_id ~value:v_id () in
                 [ y ]
             | [ q_id; k_id; v_id; m_id ] ->
                 let open Graph_builder in
                 let+ y =
                   sdpa params ~query:q_id ~key:k_id ~value:v_id ~mask:m_id ()
                 in
                 [ y ]
             | _ -> assert false))
  | _ -> None

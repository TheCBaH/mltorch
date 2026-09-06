(* Recurrent family op-dispatch arm for [Op_bridge]: `aten.lstm.input`
   (project step 14), split from op_bridge.ml. See op_bridge.ml for the
   [dispatch] facade that folds every family module (including this one)
   into one lookup.

   [hx]/[params] are ATen's own [Tensor[]] arguments -- [hx] is exactly
   [[h0; c0]], and [params] is a flat list this arm groups with
   [Lstm.Lstm.group_params] (shared with [Native_interp_lower_recurrent] so
   the two importers cannot decode the same list two different ways).
   [train]/list-arity rejections use [Lstm.Lstm.Reject], the same shared
   typed-rejection boundary [Attention.Sdpa.Reject] is for sdpa. This bridge
   always exposes all three outputs (unlike [Native_interp_lower_recurrent],
   which has a broader-graph liveness context to route a dead state output
   to [Discard]): a single isolated node has no such context, and every
   output here is a genuine, representable F32 tensor. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten.lstm.input" ->
      Some
        (let* aten_input = tensor_arg aten_env node "input" in
         let* aten_hx = tensors_arg aten_env node "hx" in
         let* aten_params = tensors_arg aten_env node "params" in
         let* has_biases = bool_arg node "has_biases" in
         let* num_layers = int_arg node "num_layers" in
         let* (_ : float) = float_arg node "dropout" in
         let* train = bool_arg node "train" in
         let* bidirectional = bool_arg node "bidirectional" in
         let* batch_first = bool_arg node "batch_first" in
         let* () =
           if train then fail (`Lstm_reject Lstm.Lstm.Reject.Train)
           else return ()
         in
         let* aten_h0, aten_c0 =
           match aten_hx with
           | [ h0; c0 ] -> return (h0, c0)
           | xs ->
               fail (`Lstm_reject (Lstm.Lstm.Reject.Hx_arity (List.length xs)))
         in
         let expected =
           Lstm.Lstm.params_length ~has_biases ~bidirectional ~num_layers
         in
         let got = List.length aten_params in
         let* () =
           if got = expected then return ()
           else
             fail
               (`Lstm_reject (Lstm.Lstm.Reject.Params_arity { expected; got }))
         in
         (* [got = expected] just above makes [group_params]'s consumption
            exact -- [params_length] and [group_params] walk the same
            layer/direction/bias structure, so this cannot raise. *)
         let grouped =
           Lstm.Lstm.group_params ~has_biases ~bidirectional ~num_layers
             aten_params
         in
         let* () = require_rank "input" ~expected:3 aten_input in
         let* () = require_rank "hx.h0" ~expected:3 aten_h0 in
         let* () = require_rank "hx.c0" ~expected:3 aten_c0 in
         let* () = require_f32 "input" aten_input in
         let* () = require_f32 "hx.h0" aten_h0 in
         let* () = require_f32 "hx.c0" aten_c0 in
         let check_direction (wih, whh, bias) =
           let* () = require_rank "params.weight_ih" ~expected:2 wih in
           let* () = require_rank "params.weight_hh" ~expected:2 whh in
           let* () = require_f32 "params.weight_ih" wih in
           let* () = require_f32 "params.weight_hh" whh in
           match bias with
           | None -> return ()
           | Some (bih, bhh) ->
               let* () = require_rank "params.bias_ih" ~expected:1 bih in
               let* () = require_rank "params.bias_hh" ~expected:1 bhh in
               let* () = require_f32 "params.bias_ih" bih in
               require_f32 "params.bias_hh" bhh
         in
         let* () =
           Err.List.iter
             (fun (fwd, rev) ->
               let* () = check_direction fwd in
               match rev with None -> return () | Some d -> check_direction d)
             grouped
         in
         (* [num_layers=0] leaves nothing to derive [hidden_size]/
            [input_size] from; [1] is an inert placeholder here -- the
            eventual [Graph_builder.lstm] call below rejects the empty
            layer list with [Shape_error.Lstm.Empty_layers] regardless of
            what these are, so no computation ever runs against them. *)
         let hidden_size, input_size =
           match grouped with
           | ((wih, whh, _), _) :: _ ->
               ((Aten_tensor.shape whh).(1), (Aten_tensor.shape wih).(1))
           | [] -> (1, 1)
         in
         let lstm_params : Lstm.Lstm.params =
           { hidden_size; input_size; batch_first }
         in
         let* x_input = native_of_aten "input" aten_input in
         let* x_h0 = native_of_aten "hx" aten_h0 in
         let* x_c0 = native_of_aten "hx" aten_c0 in
         let* xs = Err.List.map (native_of_aten "params") aten_params in
         (* [weight_ih]/[weight_hh] are rank-2 [4*hidden_size, In], landing on
            [W=rows, C=cols] under right-alignment; [bias_ih]/[bias_hh] are
            rank-1, landing on [C=4*hidden_size]. [Lstm.Lstm]'s own shapes
            put the row count on [N] for all four (matching [weight_ih]/
            [weight_hh]'s [N=4*hidden_size, C=cols] and [bias_shape]'s
            [N=4*hidden_size, C=1]), so every one of them needs a relayout
            permute -- unlike [input]/[hx], whose native shapes already
            match ATen's right-alignment positionally, regardless of
            [batch_first]. *)
         try
           build_g ~name:"lstm"
             ([ x_input; x_h0; x_c0 ] @ xs)
             (function
               | input_id :: h0_id :: c0_id :: param_ids ->
                   let open Graph_builder in
                   let grouped_ids =
                     Lstm.Lstm.group_params ~has_biases ~bidirectional
                       ~num_layers param_ids
                   in
                   let relayout_direction (wih, whh, bias) =
                     let* wih' = permute perm_lstm_weight wih in
                     let* whh' = permute perm_lstm_weight whh in
                     let* bias' =
                       match bias with
                       | None -> return None
                       | Some (bih, bhh) ->
                           let* bih' = permute perm_lstm_bias bih in
                           let* bhh' = permute perm_lstm_bias bhh in
                           return (Some (bih', bhh'))
                     in
                     return
                       {
                         Lstm.Lstm.Direction.weight_ih = wih';
                         weight_hh = whh';
                         bias = bias';
                       }
                   in
                   let rec build_layers = function
                     | [] -> return []
                     | (fwd, rev) :: rest ->
                         let* forward = relayout_direction fwd in
                         let* reverse =
                           match rev with
                           | None -> return None
                           | Some d ->
                               let* d' = relayout_direction d in
                               return (Some d')
                         in
                         let* rest' = build_layers rest in
                         return ({ Lstm.Lstm.Layer.forward; reverse } :: rest')
                   in
                   let* layers = build_layers grouped_ids in
                   let* out_id, hn_id, cn_id =
                     lstm lstm_params ~input:input_id ~layers ~h0:h0_id
                       ~c0:c0_id ()
                   in
                   return [ out_id; hn_id; cn_id ]
               | _ -> assert false)
         with Invalid_argument msg -> fail (`Validation_failure msg))
  | _ -> None

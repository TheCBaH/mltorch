(* Recurrent-family operator dispatch for [Native_interp_lower]:
   `aten.lstm.input` (project step 14). Serialized twin of [Op_bridge]'s
   own arm -- same checks, same typed rejections ([Lstm.Lstm.Reject],
   shared so the two importers cannot drift), same [Lstm.Lstm.group_params]
   grouping of the flat [params] argument. The one thing this importer has
   that the isolated-node bridge does not: a broader-graph liveness context
   ([ctx.reads]), so any of the three outputs that nothing downstream reads
   is routed to [Discard] instead of always kept live -- see
   .ai/native_multi_output_design.md. [has_biases]/[num_layers]/[train]/
   [bidirectional]/[batch_first] have no schema default, so they are
   decoded with [required_bool_arg]/[required_int_arg] rather than
   [bool_arg]/[int_arg] -- the same "no default the schema does not have"
   fix [float_arg]'s own comment gives for [eps]. *)

open Pytorch_types
open Schema_runtime
open Native_interp_decode

(* [lstm.input]'s [bias_ih]/[bias_hh]: rank-1 [4*hidden_size], landing on [C]
   under right-alignment; [Lstm.Lstm.bias_shape] carries the count on [N]
   instead (a row per gate, matching [weight_ih]/[weight_hh]'s own [N]), so
   this swaps [N]/[C] rather than reusing [perm_linear_weight] (whose [C] is
   untouched). Same permutation as [Op_bridge_decode.perm_lstm_bias]; kept as
   a separate definition since [native_interp] cannot depend on the
   ATen-linked [native_aten_bridge]. *)
let perm_lstm_bias =
  let open Axis in
  [ (N, C); (T, T); (D, D); (H, H); (W, W); (C, N) ]

let targets = [ "torch.ops.aten.lstm.input" ]

let dispatch ~ctx ~env (node : Node.t) =
  if not (List.mem node.target targets) then None
  else
    Some
      (let open Graph_builder in
       let esc = ctx.Native_interp_lower_context.esc in
       let graph = ctx.Native_interp_lower_context.graph in
       let reads = ctx.Native_interp_lower_context.reads in
       let get = Native_interp_lower_context.get ctx env node in
       match node.target with
       | "torch.ops.aten.lstm.input" ->
           let has_biases = required_bool_arg esc node "has_biases" in
           let num_layers = required_int_arg esc node "num_layers" in
           let (_ : float) = float_arg esc node "dropout" in
           let train = required_bool_arg esc node "train" in
           let bidirectional = required_bool_arg esc node "bidirectional" in
           let batch_first = required_bool_arg esc node "batch_first" in
           if train then malformed esc (`Lstm_reject Lstm.Lstm.Reject.Train);
           let h0_name, c0_name =
             match tensor_names_arg esc node "hx" with
             | [ h0; c0 ] -> (h0, c0)
             | xs ->
                 malformed esc
                   (`Lstm_reject (Lstm.Lstm.Reject.Hx_arity (List.length xs)))
           in
           let param_names = tensor_names_arg esc node "params" in
           let expected =
             Lstm.Lstm.params_length ~has_biases ~bidirectional ~num_layers
           in
           let got = List.length param_names in
           if got <> expected then
             malformed esc
               (`Lstm_reject (Lstm.Lstm.Reject.Params_arity { expected; got }));
           (* [got = expected] just above makes [group_params]'s consumption
              exact -- same invariant as [Op_bridge_recurrent]'s identical
              check, so this cannot raise. *)
           let grouped =
             Lstm.Lstm.group_params ~has_biases ~bidirectional ~num_layers
               param_names
           in
           let input_name = tensor_name esc node "input" in
           require_rank esc graph ~ssa:input_name ~role:`Lstm_input ~expected:3;
           require_rank esc graph ~ssa:h0_name ~role:`Lstm_h0 ~expected:3;
           require_rank esc graph ~ssa:c0_name ~role:`Lstm_c0 ~expected:3;
           let check_direction (wih, whh, bias) =
             require_rank esc graph ~ssa:wih ~role:`Lstm_weight_ih ~expected:2;
             require_rank esc graph ~ssa:whh ~role:`Lstm_weight_hh ~expected:2;
             match bias with
             | None -> ()
             | Some (bih, bhh) ->
                 require_rank esc graph ~ssa:bih ~role:`Lstm_bias ~expected:1;
                 require_rank esc graph ~ssa:bhh ~role:`Lstm_bias ~expected:1
           in
           List.iter
             (fun (fwd, rev) ->
               check_direction fwd;
               Option.iter check_direction rev)
             grouped;
           (* [num_layers=0] leaves nothing to derive [hidden_size]/
              [input_size] from; [1] is an inert placeholder -- see
              [Op_bridge_recurrent]'s identical comment: the eventual
              [Graph_builder.lstm] call rejects the empty layer list with
              [Shape_error.Lstm.Empty_layers] regardless of these. *)
           let hidden_size, input_size =
             match grouped with
             | ((wih, whh, _), _) :: _ ->
                 let whh_sizes =
                   static_sizes esc ~tensor:whh
                     (tensor_meta esc graph ~ssa:whh ~role:`Lstm_weight_hh)
                 in
                 let wih_sizes =
                   static_sizes esc ~tensor:wih
                     (tensor_meta esc graph ~ssa:wih ~role:`Lstm_weight_ih)
                 in
                 (List.nth whh_sizes 1, List.nth wih_sizes 1)
             | [] -> (1, 1)
           in
           let params : Lstm.Lstm.params =
             { hidden_size; input_size; batch_first }
           in
           (* [weight_ih]/[weight_hh]/[bias_ih]/[bias_hh] land on different
              axes than [Lstm.Lstm]'s own shapes expect under
              right-alignment -- same relayout [Op_bridge_recurrent] needs,
              via this module's own copies of the two permutations (this
              library cannot depend on the ATen-linked [native_aten_bridge],
              so they cannot be shared directly). [input]/[hx] need no
              permute: their native shapes already match ATen's
              right-alignment positionally, regardless of [batch_first]. *)
           let relayout_direction (wih, whh, bias) =
             let* wih' = permute perm_linear_weight (env_find esc env wih) in
             let* whh' = permute perm_linear_weight (env_find esc env whh) in
             let* bias' =
               match bias with
               | None -> return None
               | Some (bih, bhh) ->
                   let* bih' = permute perm_lstm_bias (env_find esc env bih) in
                   let* bhh' = permute perm_lstm_bias (env_find esc env bhh) in
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
           let* layers = build_layers grouped in
           let* out_id, hn_id, cn_id =
             lstm params ~input:(get "input") ~layers
               ~h0:(env_find esc env h0_name) ~c0:(env_find esc env c0_name) ()
           in
           let out_name, hn_name, cn_name =
             match output_names esc node with
             | [ o; h; c ] -> (o, h, c)
             | names ->
                 malformed esc
                   (`Output_arity
                      {
                        op = node.target;
                        serialized = List.length names;
                        derived = 3;
                      })
           in
           let discard_if_dead name id =
             if String_map.mem name (Lazy.force reads) then return ()
             else discard id
           in
           let* () = discard_if_dead out_name out_id in
           let* () = discard_if_dead hn_name hn_id in
           let* () = discard_if_dead cn_name cn_id in
           return [ out_id; hn_id; cn_id ]
       | _ -> assert false)

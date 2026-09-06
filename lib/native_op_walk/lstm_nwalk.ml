(* Assembles Lstm's walk: config space from Lstm.Lstm.Walk, recursively
   declaring one graph input per layer/direction's weight_ih/weight_hh/
   [bias_ih/bias_hh] (lstm-plan.md §2's ordering) and synthesizing a matching
   concrete tensor in the same order, so [List.combine] against
   [g.Graph_ir.Graph.inputs] pairs each declared input with its tensor. Both
   recursions are independent walks over the same [layer]/[bidirectional]
   structure -- see Lstm.Lstm.Walk.direction_shapes's comment for why this
   file, not that module, owns the graph-building half. *)

module M = struct
  module W = Lstm.Lstm.Walk (Walk_limits.L)
  include W

  type subject = Native_subject.t

  let target = "lstm"

  (* Declares one direction's inputs, in [weight_ih, weight_hh,
     [bias_ih, bias_hh]] order, returning the populated [Lstm.Lstm.Direction.t]. *)
  let build_direction ~layer ~suffix (wih_shape, whh_shape, bias_shapes) =
    let open Graph_builder in
    let* weight_ih =
      input ~shape:wih_shape ~name:(Printf.sprintf "wih%d%s" layer suffix) ()
    in
    let* weight_hh =
      input ~shape:whh_shape ~name:(Printf.sprintf "whh%d%s" layer suffix) ()
    in
    let* bias =
      match bias_shapes with
      | None -> return None
      | Some (bih_shape, bhh_shape) ->
          let* bias_ih =
            input ~shape:bih_shape
              ~name:(Printf.sprintf "bih%d%s" layer suffix)
              ()
          in
          let* bias_hh =
            input ~shape:bhh_shape
              ~name:(Printf.sprintf "bhh%d%s" layer suffix)
              ()
          in
          return (Some (bias_ih, bias_hh))
    in
    return { Lstm.Lstm.Direction.weight_ih; weight_hh; bias }

  let rec build_layers (c : W.cfg) ~layer =
    let open Graph_builder in
    if layer >= c.num_layers then return []
    else
      let shapes = W.direction_shapes c ~layer in
      let* forward = build_direction ~layer ~suffix:"" shapes in
      let* reverse =
        if c.bidirectional then
          let* d = build_direction ~layer ~suffix:"r" shapes in
          return (Some d)
        else return None
      in
      let* rest = build_layers c ~layer:(layer + 1) in
      return ({ Lstm.Lstm.Layer.forward; reverse } :: rest)

  (* Concrete tensors for one direction, in the SAME order [build_direction]
     declares its inputs. *)
  let synth_direction pcg (wih_shape, whh_shape, bias_shapes) =
    let wih, pcg = Native_tensor.synth pcg wih_shape in
    let whh, pcg = Native_tensor.synth pcg whh_shape in
    match bias_shapes with
    | None -> ([ wih; whh ], pcg)
    | Some (bih_shape, bhh_shape) ->
        let bih, pcg = Native_tensor.synth pcg bih_shape in
        let bhh, pcg = Native_tensor.synth pcg bhh_shape in
        ([ wih; whh; bih; bhh ], pcg)

  let rec synth_layers (c : W.cfg) pcg ~layer =
    if layer >= c.num_layers then ([], pcg)
    else
      let shapes = W.direction_shapes c ~layer in
      let fwd, pcg = synth_direction pcg shapes in
      let rev, pcg =
        if c.bidirectional then synth_direction pcg shapes else ([], pcg)
      in
      let rest, pcg = synth_layers c pcg ~layer:(layer + 1) in
      (fwd @ rev @ rest, pcg)

  let build pcg (c : W.cfg) =
    let input_t, pcg = Native_tensor.synth pcg (W.input_shape c) in
    let layer_tensors, pcg = synth_layers c pcg ~layer:0 in
    let h0_t, pcg = Native_tensor.synth pcg (W.state_shape c) in
    let c0_t, pcg = Native_tensor.synth pcg (W.state_shape c) in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"lstm" ~outputs:(fun (out, hn, cn) -> [ out; hn; cn ])
          @@
          let* input_id = input ~shape:(W.input_shape c) ~name:"input" () in
          let* layers = build_layers c ~layer:0 in
          let* h0_id = input ~shape:(W.state_shape c) ~name:"h0" () in
          let* c0_id = input ~shape:(W.state_shape c) ~name:"c0" () in
          lstm ~name:"out" (W.params c) ~input:input_id ~layers ~h0:h0_id
            ~c0:c0_id ())
    in
    let inputs =
      List.combine g.Graph_ir.Graph.inputs
        ([ input_t ] @ layer_tensors @ [ h0_t; c0_t ])
    in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end

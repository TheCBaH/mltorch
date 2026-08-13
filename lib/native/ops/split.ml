(* Ops that split one tensor into an ordered sequence of slices. Currently just
   [Unbind] (ATen's `aten.unbind.int`); a category file so a future
   `aten.split`/`aten.chunk` lands beside it, the way [Reduce] holds the
   reductions and [Pool] the pooling forms. *)

module Unbind = struct
  (* Selects along [axis], producing one output per coordinate of that axis:
     `unbind.int(Tensor self, int dim=0) -> Tensor[]`. Pure coordinate
     selection — no arithmetic at all. See .ai/native_graph_design.md.

     THE OUTPUT COUNT IS PART OF THE INPUT SIGNATURE, not a parameter: it is
     [extent(x_shape, axis)], so the builder derives it and callers never pass a
     second count. That is what keeps a serialized graph's SSA name list and the
     node's output arity checkable against each other rather than merely zipped.

     [axis] is a frame axis; a dispatcher gets it from ATen's rank-relative
     [dim] via [Aten_shape.axis_of_dim], so a negative or out-of-range [dim] is
     already normalized (or rejected) before it reaches this payload.

     Unbind DROPS the axis it selects along, so the survivors re-pack toward the
     inside exactly as `mean(keepdim=false)` does — same rule, same helper
     ([Aten_shape.repack_dropped]). Right-aligned [2,3,4] is [H=2 W=3 C=4];
     unbinding W gives three [W=2 C=4] outputs, and output k reads the input at
     [H=out.W, W=k, C=out.C]. Like [Mean], this needs no input rank: under the
     right-aligned embedding the axes outside the data are already extent 1, so
     shifting every surviving axis inward lands the real data correctly.

     Native tensors have no empty extent ([Dim.extent] is >= 1), so the output
     list is never empty — ATen's zero-length-dim case has no Native form and is
     refused before it reaches here. *)
  type params = { axis : Axis.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"unbind_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  (* No [map_axis] counterpart to [Reduce.Mean.map_dims], deliberately: that
     exists because [Sink_permute_mean] transports a reduction past a permute,
     and [Unbind] is NOT on the sink/reuse-permute allowlists — its semantics
     are axis-specific, so it stays a permute barrier until some pass actually
     wants to remap it. Adding the transport before that pass exists would be
     an abstraction with no caller to shape it. *)
  let pp_params fmt (p : params) = Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis.pp p.axis

  type t = { params : params; x : Tensor_ref.t }

  let name = "Unbind"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        { params = get "params" params_jsont; x = get "x" Tensor_ref.jsont })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ("x", Json_util.enc Tensor_ref.jsont t.x);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>unbind@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params

  (* Maps each surviving INPUT axis to the OUTPUT axis carrying its data. The
     selected axis is absent: it is supplied by the output ordinal instead. *)
  let kept_map (p : params) = Aten_shape.repack_dropped ~dropped:[ p.axis ]

  (* Every output has the same shape: the survivors packed right-aligned, the
     dropped axis and the vacated outer axes extent 1. *)
  let slice_shape ~(x_shape : Vec6.shape) (p : params) =
    let ones = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
    List.fold_left
      (fun s (kin, oax) -> Vec6.copy x_shape ~src:kin ~dst:oax s)
      ones (kept_map p)

  (* The one op whose output CARDINALITY is model data, so the count is an
     untrusted aggregate and has to be bounded HERE — before the list exists.
     A ceiling checked by the builder would already be too late: the builder
     asks [Graph_shape] for the shape list first, so the allocation it is meant
     to prevent has happened by the time it can measure one.

     [Kernel.Limits.Hard.outputs] is the existing cross-backend ceiling and
     [Kernel.Limits.create] applies it EXCLUSIVELY ([v >= hard]); reusing the
     constant with a different boundary rule would be worse than picking a
     second number, so 4095 is the largest accepted count. *)
  let output_limit = Kernel.Limits.Hard.outputs

  let output_shapes ~(x_shape : Vec6.shape) (p : params) =
    let count = (Vec6.get x_shape p.axis :> int) in
    if count >= output_limit then
      Err.fail
        (`Output_count_over_limit
           {
             Shape_error.Output_count.limit = output_limit;
             observed = Shape_error.Output_count.Exact count;
           })
    else
      let slice = slice_shape ~x_shape p in
      Err.return (List.init count (fun _ -> slice))

  (* This op's random-walk config space: a 4D tensor unbound along one of the
     four mutable axes. [Walk_core.Shape.t] carries no rank and fixes T/D at 1,
     so the axis is a curated field over exactly the axes that shape can vary —
     there is nothing to correlate and [cascade] is the identity. The per-axis
     caps and the numel budget already bound the output count well below
     [output_limit]. Rank/dim correlation belongs in the ATEN recipe, where the
     shape is a rank-bearing list. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; axis : Axis.t }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        axis = Axis.H;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape
    let params (c : cfg) : params = { axis = c.axis }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "axis" Axis.[ N; H; W; C ] (fun r v -> { r with axis = v });
        ]

    let pp fmt (c : cfg) =
      Format.fprintf fmt "{shape=%a axis=%a}" Walk_core.Shape.pp c.shape Axis.pp
        c.axis
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [output] is the ordinal in [Node.outputs] order, which for this op IS the
       coordinate to read at the selected axis. Unlike
       [MaxPool2dWithIndices]'s two named pixel functions, every output here runs
       the same algorithm and differs only in that constant.

       [clamp_low] (max 0) is the SOUND delta->position conversion and [output]
       is non-negative, so it is exact here. [assume_index] would also typecheck
       and is the wrong tool: its single encapsulated use is in
       [Window_axis.window], where the clip establishes the claim. *)
    let pixel (p : params) ~output ~x (out : Semantics.position S.index Vec6.t)
        =
      let zero =
        Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
          ~h:S.index_zero ~w:S.index_zero ~c:S.index_zero
      in
      (* Fold the (input axis, output axis) pairs once per pixel, exactly as
         [Reduce.Mean.Compute.pixel] does, then override the dropped axis with
         the ordinal — the coordinate no output axis carries. *)
      let base =
        List.fold_left
          (fun v (kin, oax) -> Vec6.copy out ~src:oax ~dst:kin v)
          zero (kept_map p)
      in
      S.load x (Vec6.set base p.axis (S.clamp_low (S.index_const output)))
  end
end

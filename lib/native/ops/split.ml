(* Ops that select along one named axis: [Unbind] (ATen's `aten.unbind.int`),
   which drops the axis and produces one output per coordinate of it, and
   [Slice] (`aten.slice.Tensor`), which keeps the axis and narrows it. A
   category file so a future `aten.split`/`aten.chunk` lands beside them, the
   way [Reduce] holds the reductions and [Pool] the pooling forms. *)

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

(* A strided selection along one axis, keeping the axis:
   `slice.Tensor(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None,
   SymInt step=1) -> Tensor`. Unlike [Unbind], the rank is unchanged, so there
   is no [repack_dropped] and no ordinal — one output, one axis narrowed.

   THE BOUNDS ARE CANONICAL. [start] and [stop] are non-negative, ordered, and
   within the axis; the defaulting, the negative-index normalization and the
   clamping that ATen performs are all done before a value reaches this payload,
   by [Aten_shape.resolve_slice], which both importers share so the two cannot
   drift. What is deliberately NOT delegated is the decision that the result is
   empty: that is a fact about the extent, so it belongs to [output_shape],
   where [Graph_builder] and a JSON-decoded graph meet it too.

   [step] is [Op_config.Pos.t] rather than a plain [int]: ATen requires a
   positive step ("slice step must be positive") and rejects zero and negative
   values outright, so a non-positive step is unrepresentable here instead of
   being validated at each use. *)
module Slice = struct
  type params = {
    axis : Axis.t;
    start : int;
    stop : int;
    step : Op_config.Pos.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"slice_params" (fun axis start stop step ->
        { axis; start; stop; step = Op_config.Pos.of_int step })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "start" Jsont.int ~enc:(fun p -> p.start)
    |> Jsont.Object.mem "stop" Jsont.int ~enc:(fun p -> p.stop)
    |> Jsont.Object.mem "step" Jsont.int ~enc:(fun p -> (p.step :> int))
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a start=%d stop=%d step=%d}@]" Axis.pp p.axis
      p.start p.stop
      (p.step :> int)

  type t = { params : params; x : Tensor_ref.t }

  let name = "Slice"

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
    Fmt.pf fmt "@[<hv 2>slice@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params

  (* Only the selected axis changes; every other extent is carried through, so
     the rank is preserved by construction rather than by a rule.

     Two rejections, in this order. Range FIRST, because it is what makes the
     count meaningful: with [start > stop] the ceiling below is negative and
     with [stop > extent] it is a count of elements that are not there, and
     neither is a fact worth reporting. It is also what keeps [Compute]'s read
     in bounds — [clamp_low] bounds the source coordinate below and nothing
     bounds it above.

     Every operand is already at most [Kernel.Limits.Hard.extent] once the range
     check passes ([in_extent] is a [Dim.extent], and [start]/[stop] are inside
     it), so the difference and the ceiling are computed in [int64] with no
     further bounding: on the 32-bit-[int] backends the inputs to the aggregate
     are what has to be bounded, and here the range check is that bound. *)
  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let in_extent = Vec6.get x_shape p.axis in
    let n = (in_extent :> int) in
    let step = (p.step :> int) in
    let fail fault out =
      Err.fail
        (`Slice
           Shape_error.Slice.
             {
               axis = p.axis;
               in_extent;
               start = p.start;
               stop = p.stop;
               step = p.step;
               out;
               fault;
             })
    in
    if not (0 <= p.start && p.start <= p.stop && p.stop <= n) then
      fail `Out_of_range 0L
    else
      let span = Int64.of_int (p.stop - p.start) in
      let step64 = Int64.of_int step in
      (* Ceiling division on non-negative operands, so the plain form is exact
         and [Window_axis.floor_div]'s negative-numerator correction is not
         needed — the range check above is what rules that case out. *)
      let out = Int64.div (Int64.add span (Int64.sub step64 1L)) step64 in
      if out < 1L then fail `Empty out
      else Err.return (Vec6.set x_shape p.axis (Dim.extent (Int64.to_int out)))

  (* This op's random-walk config space. The bounds are DERIVED from the drawn
     extent rather than being axes of their own: a (start, stop) pair is only
     valid for a specific extent, so independent axes would generate the empty
     and out-of-range configurations [output_shape] refuses, and a walk of
     refusals reads as coverage. The pattern names a whole intent — take the
     head, take the tail, take every second element — and [bounds] resolves it
     against whatever extent is current, which is why [cascade] is the identity.

     [Prefix] is deliberately the identity slice on any extent. It is the one
     configuration a Default-tier walk would produce, kept here so the golden
     shows it beside the configurations that actually move data. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type pattern = Prefix | Head | Tail | Interior | Stride2 | Stride3
    type cfg = { shape : Walk_core.Shape.t; axis : Axis.t; pattern : pattern }

    let initial =
      {
        shape = { Walk_core.Shape.n = 2; t = 1; d = 1; h = 4; w = 4; c = 4 };
        axis = Axis.H;
        pattern = Head;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    (* Every result is non-empty for any extent >= 1: [Head] and [Tail] keep at
       least one element because [max 1] and [min (n-1)] pin them, and the
       strided forms start at 0. *)
    let bounds ~n = function
      | Prefix -> (0, n, 1)
      | Head -> (0, max 1 (n / 2), 1)
      | Tail -> (min (n - 1) (n / 2), n, 1)
      | Interior -> (min (n - 1) 1, n, 1)
      | Stride2 -> (0, n, 2)
      | Stride3 -> (0, n, 3)

    let params (c : cfg) : params =
      let n = (Vec6.get (shape c) c.axis :> int) in
      let start, stop, step = bounds ~n c.pattern in
      { axis = c.axis; start; stop; step = Op_config.Pos.of_int step }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "axis" Axis.[ N; H; W; C ] (fun r v -> { r with axis = v });
          field_axis "pattern"
            [ Prefix; Head; Tail; Interior; Stride2; Stride3 ] (fun r v ->
              { r with pattern = v });
        ]

    let pp fmt (c : cfg) =
      let p = params c in
      Format.fprintf fmt "{shape=%a axis=%a start=%d stop=%d step=%d}"
        Walk_core.Shape.pp c.shape Axis.pp p.axis p.start p.stop
        (p.step :> int)
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    (* The whole op: one axis of the output coordinate mapped affinely back into
       the input. [clamp_low] is the SOUND delta->position conversion and is
       exact here because [start >= 0] and [step >= 1], so the value it clamps
       is already non-negative — [assume_index] would also typecheck and is the
       wrong tool, for the reason [Unbind.Compute] gives. The upper bound is
       [output_shape]'s range check, not anything here. *)
    let pixel (p : params) ~x (out : Semantics.position S.index Vec6.t) =
      let src =
        S.clamp_low
          (S.index_add (S.index_const p.start)
             (S.index_scale (p.step :> int) (S.of_index (Vec6.get out p.axis))))
      in
      S.load x (Vec6.set out p.axis src)
  end
end

(* `select.int(Tensor self, int dim, int index) -> Tensor`: narrows [axis] to
   a single position AND drops it, unlike [Slice] (which narrows but keeps the
   axis) and unlike [Unbind] (which drops every position of the axis, one
   output per coordinate, rather than one caller-chosen position). Design-goal
   fix (see .ai/native_add_op.md): this used to lower to a [Slice] node
   followed by a separate [Reshape] node at the bridge, which left no node in
   the graph naming [select.int]. Fixed by reusing [Slice]'s implementation —
   its [output_shape] and [Compute.pixel] — rather than its node, the same way
   [Unbind] reuses [Aten_shape.repack_dropped] to fold the dropped axis away.

   THE INDEX IS CANONICAL, the same discipline [Slice]'s bounds follow: ATen
   REJECTS an out-of-range index rather than clamping it (unlike [Slice]'s
   defaulted/clamped bounds), so [index] is resolved by
   [Aten_shape.resolve_index] before it reaches this payload, and is expected
   non-negative and inside the axis. Reusing [Slice.output_shape] with
   [start=index; stop=index+1] happens to enforce exactly that range
   (`0 <= start <= stop <= extent`), so no separate check is needed here — the
   op-owned check lives in [Slice], not duplicated. *)
module Select = struct
  type params = { axis : Axis.t; index : int }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"select_params" (fun axis index -> { axis; index })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "index" Jsont.int ~enc:(fun p -> p.index)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a index=%d}@]" Axis.pp p.axis p.index

  type t = { params : params; x : Tensor_ref.t }

  let name = "Select"

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
    Fmt.pf fmt "@[<hv 2>select@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params

  (* [Unbind]'s [kept_map]/[slice_shape] shape, applied to the ONE-WIDE window
     [Slice.output_shape] returns instead of to [x_shape] directly: fold the
     selected axis away by repacking every surviving axis right-aligned. *)
  let slice_params (p : params) : Slice.params =
    {
      axis = p.axis;
      start = p.index;
      stop = p.index + 1;
      step = Op_config.Pos.of_int 1;
    }

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let+ sliced = Slice.output_shape ~x_shape (slice_params p) in
    let ones = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
    List.fold_left
      (fun s (kin, oax) -> Vec6.copy sliced ~src:kin ~dst:oax s)
      ones
      (Aten_shape.repack_dropped ~dropped:[ p.axis ])

  module Compute (S : Semantics.SEMANTICS) = struct
    (* Build the SLICE-space coordinate ([Unbind.Compute]'s [base] idiom: fold
       every surviving axis inward via [kept_map], leaving the dropped axis at
       [S.index_zero] — which is exactly [Slice]'s own output coordinate at that
       axis, since the sliced window is one wide) and hand it to [Slice]'s
       [pixel] unchanged. *)
    let pixel (p : params) ~x (out : Semantics.position S.index Vec6.t) =
      let zero =
        Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
          ~h:S.index_zero ~w:S.index_zero ~c:S.index_zero
      in
      let base =
        List.fold_left
          (fun v (kin, oax) -> Vec6.copy out ~src:oax ~dst:kin v)
          zero
          (Aten_shape.repack_dropped ~dropped:[ p.axis ])
      in
      let module SliceC = Slice.Compute (S) in
      SliceC.pixel (slice_params p) ~x base
  end
end

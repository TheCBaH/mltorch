(* The selection/joining/splitting family of [Ops4] payloads, split out of
   ops4.ml under the tracked file-size ceiling (scripts/check-file-size.sh).
   [Ops4] re-exports every module below by the same name, so external
   references such as [Ops4.Slice4] or [Ops4.Unbind] are unaffected. *)

(* ---- selection ------------------------------------------------------------ *)

(* Native's [Slice] with its axis narrowed to the dialect's four. Its own payload
   for the FIRST reason in .ai/native4d_add_op.md -- it names an axis, so the
   field is [Axis4.t] and T/D are unsayable in a four-axis graph.

   The BOUNDS are Native's own and cross unchanged: they are canonical by the
   time any payload exists, and nothing about narrowing the axis set changes
   what [0 <= start <= stop <= extent] means. Restating the shape rule or the
   pixel map here would be a second definition free to drift from the [Compute]
   that reads them, so both are delegated to [Split.Slice]. *)
module Slice4 = struct
  type params = {
    axis : Axis4.t;
    start : int;
    stop : int;
    step : Op_config.Pos.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"slice4_params" (fun axis start stop step ->
        { axis; start; stop; step = Op_config.Pos.of_int step })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "start" Jsont.int ~enc:(fun p -> p.start)
    |> Jsont.Object.mem "stop" Jsont.int ~enc:(fun p -> p.stop)
    |> Jsont.Object.mem "step" Jsont.int ~enc:(fun p -> (p.step :> int))
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a start=%d stop=%d step=%d}@]" Axis4.pp p.axis
      p.start p.stop
      (p.step :> int)

  type t = { params : params; x : Tensor_ref.t }

  let name = "Slice4"

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
    Fmt.pf fmt "@[<hv 2>slice4@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params
end

(* Native's [Split.Select] with its axis narrowed to the dialect's four. Its own
   payload for the FIRST reason in .ai/native4d_add_op.md -- it names an axis,
   so the field is [Axis4.t] and T/D are unsayable in a four-axis graph.

   Unlike [Slice4], which KEEPS its axis, [Select4] DROPS it, the same way
   [Unbind] does -- and the representable set is narrower for the same reason
   [Unbind]'s own comment gives: dropping an axis shifts every axis OUTSIDE it
   (toward N) one place further out, so a four-axis source stays four-axis only
   when whatever lands on T is unit. [Shape4.of_vec6] on the inferred output is
   what the lowerer uses to catch that, not a rule stated here.

   THE INDEX IS CANONICAL, the same discipline [Slice4]'s bounds follow: ATen
   rejects an out-of-range index rather than clamping it, so [index] is already
   resolved against the axis extent by the time it reaches this payload.
   [index] itself crosses unchanged, and the shape rule / pixel map delegate to
   [Split.Select] -- which itself delegates to [Split.Slice] -- rather than
   restating that composition here. *)
module Select4 = struct
  type params = { axis : Axis4.t; index : int }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"select4_params" (fun axis index -> { axis; index })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "index" Jsont.int ~enc:(fun p -> p.index)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a index=%d}@]" Axis4.pp p.axis p.index

  type t = { params : params; x : Tensor_ref.t }

  let name = "Select4"

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
    Fmt.pf fmt "@[<hv 2>select4@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params
end

(* The write-back counterpart of [Select4]: [self] with position [index] of
   [axis] replaced by [src] (which has [Select4]'s own output shape) and
   every other position carried through unchanged. Its own payload for the
   same reason [Select4]'s is -- it names an axis, so the field is [Axis4.t]
   -- and reuses [Split.Select_scatter]'s shape rule/pixel map rather than
   restating them, the same delegation [Select4] makes to [Split.Select]. *)
module Select_scatter4 = struct
  type params = { axis : Axis4.t; index : int }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"select_scatter4_params" (fun axis index ->
        { axis; index })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "index" Jsont.int ~enc:(fun p -> p.index)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a index=%d}@]" Axis4.pp p.axis p.index

  type t = { params : params; self : Tensor_ref.t; src : Tensor_ref.t }

  let name = "Select_scatter4"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          self = get "self" Tensor_ref.jsont;
          src = get "src" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ("self", Json_util.enc Tensor_ref.jsont t.self);
            ("src", Json_util.enc Tensor_ref.jsont t.src);
          ])
      Jsont.json

  let operands (t : t) = [ t.self; t.src ]
  let map_operands f (t : t) = { t with self = f t.self; src = f t.src }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>select_scatter4@ self=%a@ src=%a@ params=%a@]" pp_ref
      t.self pp_ref t.src pp_params t.params
end

(* ---- joining ---------------------------------------------------------------

   Native's [Concat] with its axis narrowed to the dialect's four. Its own
   payload for the FIRST reason in .ai/native4d_add_op.md -- it names an axis,
   so the field is [Axis4.t] and T/D are unsayable in a four-axis graph.

   [xs] is a variadic OPERAND list, unlike every other op above except
   [Layer_norm]/[Depthwise_conv2d]'s fixed few -- the same "operand count is
   not just output count" case Native's own [Concat] documents. Nothing here
   restates the shape rule or the pixel map: both delegate to [Concat.Concat],
   which is what keeps them from drifting apart. *)
module Concat4 = struct
  type params = { axis : Axis4.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"concat4_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis4.pp p.axis

  type t = { params : params; xs : Tensor_ref.t list }

  let name = "Concat4"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          xs = get "xs" (Jsont.list Tensor_ref.jsont);
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ( "xs",
              Json_util.jarr (List.map (Json_util.enc Tensor_ref.jsont) t.xs) );
          ])
      Jsont.json

  let operands (t : t) = t.xs
  let map_operands f (t : t) = { t with xs = List.map f t.xs }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>concat4@ xs=%a@ params=%a@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_ref))
      t.xs pp_params t.params
end

(* ---- stacking --------------------------------------------------------------

   Native's [Concat.Stack] with its axis narrowed to the dialect's four. Its
   own payload for the FIRST reason in .ai/native4d_add_op.md -- it names an
   axis, so the field is [Axis4.t] and T/D are unsayable in a four-axis graph.

   [Stack] INSERTS a new axis, the opposite of [Select4]'s drop -- so the same
   shift argument applies in reverse: inserting at [axis] pushes every axis
   OUTSIDE it (toward N) one place further out, and a four-axis result stays
   four-axis only when whatever lands on D is unit. [Shape4.of_vec6] on the
   inferred output is what the lowerer uses to catch that, not a rule stated
   here -- exactly [Select4]'s own discipline, mirrored.

   [xs] is a variadic OPERAND list like [Concat4]'s; nothing here restates the
   shape rule or the pixel map, both delegate to [Concat.Stack] (itself built
   from [Concat.Concat] over each operand's unsqueezed shape), which is what
   keeps them from drifting apart. *)
module Stack4 = struct
  type params = { axis : Axis4.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"stack4_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis4.pp p.axis

  type t = { params : params; xs : Tensor_ref.t list }

  let name = "Stack4"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          xs = get "xs" (Jsont.list Tensor_ref.jsont);
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ( "xs",
              Json_util.jarr (List.map (Json_util.enc Tensor_ref.jsont) t.xs) );
          ])
      Jsont.json

  let operands (t : t) = t.xs
  let map_operands f (t : t) = { t with xs = List.map f t.xs }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>stack4@ xs=%a@ params=%a@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_ref))
      t.xs pp_params t.params
end

(* ---- splitting ------------------------------------------------------------ *)

(* The dialect's first MULTI-OUTPUT op: one output per coordinate of [axis], so
   its arity comes from the operand's signature rather than from the op.

   Needs its own payload for the first of the three reasons in
   .ai/native4d_add_op.md — it NAMES AN AXIS, so the field is [Axis4.t] and T/D
   are unsayable. Everything else is Native's [Split.Unbind]: the shape rule and
   the per-pixel algorithm are delegated, never restated.

   The representable set is narrower than it looks, and [Shape4.of_vec6] is what
   enforces it rather than any rule written here. Dropping an axis shifts every
   axis OUTSIDE it one place outward, so a four-axis [N,H,W,C] input unbound
   along H/W/C puts N's extent on T. That is inside the dialect only when N=1 —
   a rank-3 source is the common way to get there, but any batch-1 input
   qualifies. Unbinding N always converts, since nothing is outside it. *)
module Unbind = struct
  type params = { axis : Axis4.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"unbind_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis4.pp p.axis

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
end

(* [Unbind]'s rank-preserving sibling: KEEPS [axis], dividing it into
   contiguous windows of the given [sizes] instead of dropping it entirely.
   The arity comes from [sizes]'s length, a parameter, unlike [Unbind]'s
   (derived from the operand's signature) -- the same difference as Native's
   own [Split.Split_with_sizes] versus [Split.Unbind].

   Needs its own payload for the same reason [Unbind] does -- it NAMES AN
   AXIS -- and for no other: [sizes] is an ordinary int list, carrying no
   shape and naming no axis, so it crosses unchanged the way [Pad4]'s signed
   amounts do. The shape rule and the per-pixel offset are Native's
   [Split.Split_with_sizes], delegated rather than restated.

   Unlike [Unbind], every output has the SAME axis-domain answer as the
   input: keeping [axis] means no axis shifts, so a four-axis source stays
   four-axis in every slice regardless of which axis is split or how many
   pieces. [Shape4.of_vec6] still re-checks each inferred slice (the lowerer
   is not exempt from proving what it emits), but there is no N=1-shaped
   precondition here the way there is for [Unbind]. *)
module Split_with_sizes4 = struct
  type params = { axis : Axis4.t; sizes : int list }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"split_with_sizes4_params" (fun axis sizes ->
        { axis; sizes })
    |> Jsont.Object.mem "axis" Axis4.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "sizes" (Jsont.list Jsont.int) ~enc:(fun p -> p.sizes)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a sizes=%a}@]" Axis4.pp p.axis
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma Fmt.int))
      p.sizes

  type t = { params : params; x : Tensor_ref.t }

  let name = "SplitWithSizes4"

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
    Fmt.pf fmt "@[<hv 2>split_with_sizes4@ x=%a@ params=%a@]" pp_ref t.x
      pp_params t.params
end

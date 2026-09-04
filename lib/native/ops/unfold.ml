(* `aten.unfold.default(Tensor self, int dimension, int size, int step) ->
   Tensor`: a sliding, possibly-overlapping window view along ONE axis. The
   named axis keeps its identity but shrinks to the window COUNT
   `(extent - size) / step + 1`; a genuinely NEW axis of extent `size` is
   appended holding each window's own elements. Real data movement (windows
   overlap when `step < size`), not a reshape.

   Native has no rank: every tensor is always six axes, right-aligned per
   ATen rank by `Aten_shape.of_aten`. Appending an ATen axis therefore
   RELABELS every axis already in the frame -- of_aten's own rule right-
   aligns a rank-(r+1) tensor one slot further left than a rank-r one (see
   aten_shape.ml). Concretely, per `Axis.all = [N;T;D;H;W;C]`: content that
   was on `T` moves to `N`, `D` moves to `T`, `H` moves to `D`, `W` moves to
   `H`, `C` moves to `W`, and the fresh window axis always lands on `C` --
   the newly vacated last slot. Old `N` has nowhere to go, so it must
   already be unit (ATen rank <= 5); anything else means the frame has no
   room left for the axis this op introduces. Hand-verified against the
   corpus's own two-chained-unfold HaloAttn case (each occupied frame axis
   checked individually, not just the shape) before trusting this rule --
   see the landing note.

   [params.axis] therefore names the OUTPUT axis carrying the window count --
   never `C`, which this op always reserves for the window offset -- and the
   op needs no explicit ATen rank: [source_of]/[carried_from] below express
   the whole relabeling as one fixed rotation over [Axis.t], with the single
   precondition ([x_shape]'s own `N` is unit) standing in for "rank <= 5". *)

module Unfold = struct
  type params = {
    axis : Axis.t;
    size : Dim.extent Dim.t;
    step : Op_config.Pos.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"unfold_params" (fun axis size step ->
        { axis; size; step })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "size" Dim.extent_jsont ~enc:(fun p -> p.size)
    |> Jsont.Object.mem "step" Op_config.Pos.jsont ~enc:(fun p -> p.step)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a;@ size=%a;@ step=%a}@]" Axis.pp p.axis Dim.pp
      p.size Op_config.Pos.pp p.step

  type t = { params : params; x : Tensor_ref.t }

  let name = "Unfold"

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
    Fmt.pf fmt "@[<hv 2>unfold@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params

  (* The OLD (pre-unfold) axis whose content lands on NEW axis [a], for every
     [a] except [C] (always the fresh window offset). One step toward [C] in
     [Axis.all]'s fixed order -- total on {N,T,D,H,W}, the only domain it is
     ever called on. *)
  let source_of = function
    | Axis.N -> Axis.T
    | Axis.T -> Axis.D
    | Axis.D -> Axis.H
    | Axis.H -> Axis.W
    | Axis.W -> Axis.C
    | Axis.C -> invalid_arg "Unfold.source_of: C has no source axis"

  (* Inverse of [source_of]: the NEW axis that OLD axis [o] shifts onto, for
     every [o] except [N] (never a shift target -- its content is discarded,
     which is safe only because [output_shape] required it to be unit). *)
  let dest_of = function
    | Axis.T -> Axis.N
    | Axis.D -> Axis.T
    | Axis.H -> Axis.D
    | Axis.W -> Axis.H
    | Axis.C -> Axis.W
    | Axis.N -> invalid_arg "Unfold.dest_of: N has no destination axis"

  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    if Axis.equal p.axis Axis.C then
      Err.fail
        (`Unfold Shape_error.Unfold.{ axis = p.axis; fault = Reserved_axis })
    else if not (Dim.equal (Vec6.get x_shape Axis.N) (Dim.extent 1)) then
      Err.fail
        (`Unfold Shape_error.Unfold.{ axis = p.axis; fault = Rank_exceeded })
    else
      let source = source_of p.axis in
      let+ count =
        Window_axis.output_extent ~ceil_mode:false ~kernel:p.size ~stride:p.step
          ~pad_before:(Op_config.Nonneg.of_int 0)
          ~pad_after:(Op_config.Nonneg.of_int 0)
          ~dilation:(Op_config.Pos.of_int 1)
          ~in_extent:(Vec6.get x_shape source)
      in
      let carried a =
        if Axis.equal a p.axis then count else Vec6.get x_shape (source_of a)
      in
      Vec6.make ~n:(carried Axis.N) ~t:(carried Axis.T) ~d:(carried Axis.D)
        ~h:(carried Axis.H) ~w:(carried Axis.W) ~c:p.size

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~x (out : Semantics.position S.index Vec6.t) =
      let source = source_of p.axis in
      let widx = Vec6.get out p.axis in
      let k = Vec6.get out Axis.C in
      (* In bounds by construction: [output_shape]'s [count] guarantees
         [(count-1)*step + (size-1) < in_extent] for every valid [widx]/[k],
         the same invariant [Window_axis.output_extent] gives conv's own
         clipped window -- except here every window is fully interior, so no
         clamping is needed at all. *)
      let windowed =
        S.assume_index
          (S.index_add
             (S.index_scale (p.step :> int) (S.of_index widx))
             (S.of_index k))
      in
      let old o =
        if Axis.equal o source then windowed else Vec6.get out (dest_of o)
      in
      S.load6 x ~n:S.index_zero ~t:(old Axis.T) ~d:(old Axis.D) ~h:(old Axis.H)
        ~w:(old Axis.W) ~c:(old Axis.C)
  end
end

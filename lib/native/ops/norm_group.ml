(* Group normalisation (ATen `aten.group_norm`) -- split out of the [Norm]
   category file ([norm.ml]) under the tracked file-size ceiling. Shares
   [normalized_shape] with [Norm_rms]/[Norm_layer] via [Norm_shared]. *)

open Norm_shared

module GroupNorm = struct
  (* Group normalisation (ATen `aten.group_norm`). Reshapes [channel]'s extent
     into [groups] equal-sized chunks and normalises each (N, group) slice
     over that chunk together with every axis but [N] and [channel] -- T, D,
     H, W in the always-six-axis frame, not a caller-chosen [dims] list the
     way [LayerNorm]/[RmsNorm] take one, since ATen's group_norm always
     reduces over "everything except batch and (the group's slice of)
     channel". Unlike them, [weight]/[bias] are full PER-CHANNEL vectors
     (ATen's group_norm affine has shape [C], not the normalized-shape
     [RmsNorm]/[LayerNorm] carry) -- read at the output pixel's own channel
     index and broadcast elsewhere, the same [BatchNorm.Compute] pattern.

       count = channels_per_group * (product of every non-N/channel extent)
       group = floor(channel_index / channels_per_group)
       mean  = sum over that group's window / count
       var   = sum over (x - mean)^2 in that window / count
       y     = (x - mean) / sqrt(var + eps) * weight[channel] + bias[channel]

     [channel] is a parameter (mirroring [BatchNorm.params.channel]) rather
     than hardcoded to [Axis.C], though every importer sets it there today. *)
  type params = { channel : Axis.t; groups : Op_config.Pos.t; eps : float }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"group_norm_params" (fun channel groups eps ->
        { channel; groups = Op_config.Pos.of_int groups; eps })
    |> Jsont.Object.mem "channel" Axis.jsont ~enc:(fun p -> p.channel)
    |> Jsont.Object.mem "groups" Jsont.int ~enc:(fun p -> (p.groups :> int))
    |> Jsont.Object.mem "eps" Json_util.f32_jsont ~enc:(fun p -> p.eps)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{channel=%a;@ groups=%d;@ eps=%a}@]" Axis.pp p.channel
      (p.groups :> int)
      Fmt.float p.eps

  type t = {
    params : params;
    x : Tensor_ref.t;
    weight : Tensor_ref.t option;
    bias : Tensor_ref.t option;
  }

  let name = "Group_norm"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          x = get "x" Tensor_ref.jsont;
          weight = Json_util.opt_field ms "weight" Tensor_ref.jsont;
          bias = Json_util.opt_field ms "bias" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        let opt k = function None -> [] | Some r -> [ (k, ref_ r) ] in
        Json_util.jobj
          ([ ("params", Json_util.enc params_jsont t.params) ]
          @ opt "weight" t.weight @ opt "bias" t.bias
          @ [ ("x", ref_ t.x) ]))
      Jsont.json

  let operands (t : t) =
    (t.x :: Option.to_list t.weight) @ Option.to_list t.bias

  let map_operands f (t : t) =
    {
      t with
      x = f t.x;
      weight = Option.map f t.weight;
      bias = Option.map f t.bias;
    }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>group_norm@ x=%a@ weight=%a@ bias=%a@ params=%a@]"
      pp_ref t.x
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.weight
      (Fmt.option ~none:(Fmt.any "none") pp_ref)
      t.bias pp_params t.params

  (* Every axis except N and [channel]: what a group reduces over besides its
     own channel slice. Not a caller [dims] list -- ATen's group_norm has no
     axis-selection parameter, so this is fixed by the op's own semantics. *)
  let spatial_dims (channel : Axis.t) =
    List.filter
      (fun a -> (not (Axis.equal a Axis.N)) && not (Axis.equal a channel))
      Axis.all

  (* The shape one group reduces over: every [spatial_dims] axis at its full
     extent, [channel] narrowed to one group's slice, every other axis extent
     1. Used only to BOUND the reduction count -- [Vec6.numel_bounded] divides
     each factor into the ceiling before multiplying, the aggregate rule
     CLAUDE.md states and [normalized_count] above already follows. *)
  let reduce_shape ~(x_shape : Vec6.shape) ~(channel : Axis.t)
      ~(channels_per_group : Dim.extent Dim.t) =
    let base =
      List.fold_left
        (fun acc a -> Vec6.set acc a (Vec6.get x_shape a))
        (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
        (spatial_dims channel)
    in
    Vec6.set base channel channels_per_group

  (* Output keeps the input shape: group norm rescales, it does not reduce.
     [num_groups] must divide the channel count exactly -- ATen's own
     contract, checked here since it arrives as an ordinary int from a PT2
     graph rather than a value ATen has already validated -- and the
     reduction count is bounded the same way [RmsNorm]/[LayerNorm]'s are. *)
  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let channels = Vec6.get x_shape p.channel in
    let groups = (p.groups :> int) in
    if (channels :> int) mod groups <> 0 then
      Err.fail
        (`Group_norm Shape_error.Group_norm.{ channels; groups = p.groups })
    else
      let cpg = Dim.extent ((channels :> int) / groups) in
      let+ (_ : int64) =
        Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel
          (reduce_shape ~x_shape ~channel:p.channel ~channels_per_group:cpg)
      in
      x_shape

  (* The same product as an [int], for [Compute], which has no error channel.
     Sound as a plain fold ONLY because [output_shape] has already run
     [Vec6.numel_bounded] on the identical product ([reduce_shape]'s numel) --
     the same precondition [normalized_count_unchecked] documents above, one
     level up: a node whose shape rule never ran cannot be evaluated. *)
  let count_unchecked ~(x_shape : Vec6.shape) ~(channel : Axis.t) ~cpg =
    List.fold_left
      (fun acc a -> acc * (Vec6.get x_shape a :> int))
      cpg (spatial_dims channel)

  (* Both affine operands are full PER-CHANNEL vectors -- [normalized_shape]
     with a single-axis [dims] happens to be exactly that layout ([channel] at
     its full extent, every other axis 1), so this reuses it rather than
     restating it; checked for the reason [LayerNorm.check_affine]'s doc
     comment gives -- an operand of the wrong extent otherwise builds a graph
     and raises out of [Tensor.read]'s bounds check partway through the
     result. *)
  let check_affine ~(x_shape : Vec6.shape) ~(channel : Axis.t)
      ~(weight : Vec6.shape option) ~(bias : Vec6.shape option) :
      (unit, Shape_error.t) Err.t =
    let expected = normalized_shape ~x_shape ~dims:[ channel ] in
    let check operand = function
      | None -> Err.return ()
      | Some actual ->
          if
            List.for_all
              (fun a -> Dim.equal (Vec6.get expected a) (Vec6.get actual a))
              Axis.all
          then Err.return ()
          else
            Err.fail
              (`Operand_shape
                 Shape_error.Operand_shape.{ operand; expected; actual })
    in
    let open Err.Syntax in
    let* () = check `Group_norm_weight weight in
    check `Group_norm_bias bias

  (* Walk config: [groups] must divide the channel extent, so it is derived
     from the drawn shape's C via [divisor] rather than an independent axis --
     the same "derive from the drawn extent" shape [Slice.Walk]'s [bounds] and
     [Split_with_sizes.Walk]'s [sizes_for] use, and for the same reason: an
     independent [groups] axis would draw far more configurations
     [output_shape] refuses (indivisible) than ones it accepts. The three
     candidates span group_norm's range: 1 (whole-channel), 2, and C itself
     (per-channel groups -- group_norm's instance-norm corner). *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = {
      shape : Walk_core.Shape.t;
      divisor : int;
      eps : float;
      weight : bool;
      bias : bool;
    }

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 3; w = 3; c = 6 };
        divisor = 1;
        eps = 1e-5;
        weight = true;
        bias = true;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    let groups (c : cfg) =
      let c_extent = (Vec6.get (shape c) Axis.C :> int) in
      let candidates =
        List.filter (fun g -> c_extent mod g = 0) [ 1; 2; c_extent ]
      in
      List.nth candidates (c.divisor mod List.length candidates)

    let weight_present (c : cfg) = c.weight
    let bias_present (c : cfg) = c.bias

    let params (c : cfg) : params =
      {
        channel = Axis.C;
        groups = Op_config.Pos.of_int (groups c);
        eps = c.eps;
      }

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "divisor" [ 0; 1; 2 ] (fun c v -> { c with divisor = v });
          field_axis "eps" [ 1e-5; 0.; 1e-3 ] (fun c v -> { c with eps = v });
          field_axis "weight" [ true; false ] (fun c v -> { c with weight = v });
          field_axis "bias" [ true; false ] (fun c v -> { c with bias = v });
        ]

    let pp fmt (c : cfg) =
      let p = params c in
      Format.fprintf fmt "{shape=%a groups=%d eps=%g weight=%b bias=%b}"
        Walk_core.Shape.pp c.shape
        (p.groups :> int)
        c.eps c.weight c.bias
  end

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x ~weight ~bias
        (out : Semantics.position S.index Vec6.t) =
      let channel = p.channel in
      let c_extent = (Vec6.get x_shape channel :> int) in
      let groups = (p.groups :> int) in
      let cpg = c_extent / groups in
      let cpg_pos = Op_config.Pos.of_int cpg in
      (* Which group [out]'s channel coordinate falls in, and that group's
         window [lo, hi) on [channel] -- [lo] is [group * cpg], always
         non-negative since both factors are, so [clamp_low] is the SOUND
         delta->position conversion the same way every other windowed-offset
         site in this engine uses it. *)
      let group =
        S.index_floor_div_pos (S.of_index (Vec6.get out channel)) cpg_pos
      in
      let lo = S.clamp_low (S.index_scale cpg group) in
      let hi = S.index_add (S.of_index lo) (S.index_const cpg) in
      let zero = Vec6.map (fun _ -> S.index_zero) out in
      (* One reduction nest over [channel]'s WINDOW plus every spatial axis's
         FULL extent -- mirrors [LayerNorm.Legacy_pixel]'s [sum_over], generalised
         to a non-zero lower bound on the one axis that needs it. *)
      let sum_over leaf =
        let rec go dims override =
          match dims with
          | [] ->
              let idx =
                List.fold_left (fun v (a, i) -> Vec6.set v a i) out override
              in
              leaf (S.load x idx)
          | d :: rest when Axis.equal d channel ->
              S.sum ~lo ~hi (fun i -> go rest ((d, i) :: override))
          | d :: rest ->
              S.sum ~lo:S.index_zero
                ~hi:(S.index_extent (Vec6.get x_shape d))
                (fun i -> go rest ((d, i) :: override))
        in
        go (channel :: spatial_dims channel) []
      in
      let count =
        S.const (float_of_int (count_unchecked ~x_shape ~channel ~cpg))
      in
      let mean = S.div (sum_over (fun v -> v)) count in
      let var =
        S.div
          (sum_over (fun v ->
               let d = S.sub v mean in
               S.mul d d))
          count
      in
      let inv = S.div (S.const 1.) (S.sqrt (S.add var (S.const p.eps))) in
      let at_channel v = S.load v (Vec6.copy_axis out channel zero) in
      S.add
        (S.mul (S.mul (S.sub (S.load x out) mean) inv) (at_channel weight))
        (at_channel bias)
  end
end

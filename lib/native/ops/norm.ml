(* Normalisations over a set of axes (NHWC). Currently just [RmsNorm] (ATen's
   `aten.rms_norm`); a category file so a future `LayerNorm` lands beside it, the
   way [Reduce] holds the reductions. *)

module RmsNorm = struct
  (* Root-mean-square normalisation (ATen `aten.rms_norm`). Unlike a [Reduce],
     the output keeps the input's full shape: the reduction over [dims] only
     computes the per-(non-normalised-coordinate) scale, which then divides every
     element. For each output pixel,

       y = x / sqrt(mean_{dims}(x^2) + eps) * weight

     where the mean of squares is taken over the normalised axes [dims] at the
     pixel's fixed non-normalised coordinates. [weight] carries the
     normalised_shape (ATen indexes it by the trailing/normalised dims only), so
     it is read at the output coordinate on each [dims] axis and broadcast
     (size-1, index 0) on every other axis. See .ai/native_compute_design.md §2.

     [dims] is a list of frame axes (a dispatcher gets them from the ATen
     normalized_shape via [Aten_shape.axis_of_dim]); one op covers
     `rms_norm(normalized_shape=[C])` and the multi-axis case alike. *)
  type params = { dims : Axis.t list; eps : float }

  (* Output keeps the input shape: rms-norm rescales, it does not reduce. *)
  let output_shape ~(x_shape : Vec6.shape) = x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(x_shape : Vec6.shape) ~x ~weight
        (out : Axis.t -> Semantics.position S.index) =
      (* Sum of x^2 over the normalised axes at the fixed non-normalised coords
         [out]: nest one [sum] per reduced axis (full extent), squaring the read
         at the leaf. Mirrors [Reduce.Mean]'s reduction nest. *)
      let rec sum_sq dims override =
        match dims with
        | [] ->
            let v =
              S.load x (fun a ->
                  match List.assoc_opt a override with
                  | Some i -> i
                  | None -> out a)
            in
            S.mul v v
        | d :: rest ->
            S.sum ~lo:S.index_zero
              ~hi:(S.index_extent (Vec6.get x_shape d))
              (fun i -> sum_sq rest ((d, i) :: override))
      in
      let count =
        List.fold_left (fun acc d -> acc * (Vec6.get x_shape d :> int)) 1 p.dims
      in
      let ms = S.div (sum_sq p.dims []) (S.const (float_of_int count)) in
      (* 1 / sqrt(mean(x^2) + eps) — ATen's rsqrt, expressed with the sqrt
         primitive. *)
      let inv = S.div (S.const 1.) (S.sqrt (S.add ms (S.const p.eps))) in
      (* weight is indexed by the normalised axes only (size 1 elsewhere). *)
      let w =
        S.load weight (fun a ->
            if List.mem a p.dims then out a else S.index_zero)
      in
      S.mul (S.mul (S.load x out) inv) w
  end
end

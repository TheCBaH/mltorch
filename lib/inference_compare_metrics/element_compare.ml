open Inference_compare_schema
module Codecs = Inference_compare_codecs
module E = Codecs.Element_codec
module L = Codecs.Logical_value
module F = Float_ops

type error =
  | Codec of Codecs.Codec_error.t
  | Policy_codec_mismatch
  | Spec_mismatch

let pp_error ppf = function
  | Codec c -> Codecs.Codec_error.pp ppf c
  | Policy_codec_mismatch ->
      Fmt.string ppf "element policy does not fit the codec"
  | Spec_mismatch -> Fmt.string ppf "left and right specs differ"

let fail e = Err.fail ~pp_error e

let rule_passed (r : Element_result.t) =
  Option.map (fun v -> Int64.equal v 0L) r.tolerance_violations

let i64 = Int64.of_int

(* ---- statistics ---------------------------------------------------------- *)

module Stats = struct
  type extremum = {
    mutable flat : int;
    mutable value : float;
    mutable set : bool;
  }

  type t = {
    direction : Difference_statistics.direction;
    mutable finite_pairs : int;
    mutable excluded : int;
    mutable positive : int;
    mutable negative : int;
    mutable zero : int;
    abs_sum : Accum.Sum.t;
    squares : Accum.Sum_squares.t;
    welford : Accum.Welford.t;
    mutable overflow : bool;
    minimum : extremum;
    maximum : extremum;
    largest : extremum;
  }

  let ext () = { flat = 0; value = 0.0; set = false }

  let create direction =
    {
      direction;
      finite_pairs = 0;
      excluded = 0;
      positive = 0;
      negative = 0;
      zero = 0;
      abs_sum = Accum.Sum.create ();
      squares = Accum.Sum_squares.create ();
      welford = Accum.Welford.create ();
      overflow = false;
      minimum = ext ();
      maximum = ext ();
      largest = ext ();
    }

  let residual t a b =
    match t.direction with
    | Right_minus_left -> b -. a
    | Left_minus_right -> a -. b

  (* Strict comparisons keep the first row-major coordinate on ties. *)
  let consider ext flat v better =
    if (not ext.set) || better v ext.value then (
      ext.set <- true;
      ext.flat <- flat;
      ext.value <- v)

  let add_finite t flat a b =
    t.finite_pairs <- t.finite_pairs + 1;
    let e = residual t a b in
    if not (Float.is_finite e) then t.overflow <- true
    else (
      if e > 0.0 then t.positive <- t.positive + 1
      else if e < 0.0 then t.negative <- t.negative + 1
      else t.zero <- t.zero + 1;
      Accum.Sum.add t.abs_sum (Float.abs e);
      Accum.Sum_squares.add t.squares e;
      Accum.Welford.add t.welford e;
      consider t.minimum flat e ( < );
      consider t.maximum flat e ( > );
      consider t.largest flat (Float.abs e) ( > ))

  let finish (t : t) ~(variance : Difference_statistics.variance) shape :
      Difference_statistics.t =
    let open Difference_statistics in
    let n = float_of_int t.finite_pairs in
    let overflowed =
      t.overflow
      || Accum.Sum.overflowed t.abs_sum
      || Accum.Welford.overflowed t.welford
    in
    let unavailable u : scalar = Unavailable u in
    let none = unavailable Empty_finite_set in
    let ext_opt e =
      if overflowed || not e.set then None
      else Some { index = F.unravel shape e.flat; value = Finite_value e.value }
    in
    let scalar f : scalar =
      if overflowed then unavailable Numeric_overflow
      else if t.finite_pairs = 0 then none
      else
        let v = f () in
        if Float.is_finite v then Finite_value v
        else unavailable Numeric_overflow
    in
    {
      status = (if overflowed then Overflowed else Computed);
      finite_pairs = i64 t.finite_pairs;
      excluded_nonfinite_pairs = i64 t.excluded;
      positive_errors = i64 t.positive;
      negative_errors = i64 t.negative;
      zero_errors = i64 t.zero;
      minimum_signed_error = ext_opt t.minimum;
      maximum_signed_error = ext_opt t.maximum;
      largest_absolute_error = ext_opt t.largest;
      bias = scalar (fun () -> Accum.Welford.mean t.welford);
      standard_deviation =
        (if overflowed then unavailable Numeric_overflow
         else if t.finite_pairs = 0 then none
         else
           match
             Accum.Welford.deviation t.welford ~sample:(variance = Sample)
           with
           | None -> unavailable Insufficient_samples
           | Some v ->
               if Float.is_finite v then Finite_value v
               else unavailable Numeric_overflow);
      mean_absolute_error = scalar (fun () -> Accum.Sum.value t.abs_sum /. n);
      root_mean_square_error =
        scalar (fun () -> Accum.Sum_squares.root_mean t.squares n);
    }
end

(* ---- magnitude summaries ------------------------------------------------- *)

module Mags = struct
  type t = {
    mutable finite : int;
    mutable zero_over_zero : int;
    mutable unbounded : int;
    mutable nonfinite : int;
    mutable overflow : int;
    mutable min : float;
    mutable max : float;
    mutable max_flat : int;
    welford : Accum.Welford.t;
  }

  let create () =
    {
      finite = 0;
      zero_over_zero = 0;
      unbounded = 0;
      nonfinite = 0;
      overflow = 0;
      min = Float.infinity;
      max = Float.neg_infinity;
      max_flat = 0;
      welford = Accum.Welford.create ();
    }

  let add t flat (m : Difference_measurement.magnitude) =
    match m with
    | Finite v ->
        t.finite <- t.finite + 1;
        if v < t.min then t.min <- v;
        if v > t.max then (
          t.max <- v;
          t.max_flat <- flat);
        Accum.Welford.add t.welford v
    | Unbounded -> t.unbounded <- t.unbounded + 1
    | Unavailable Zero_over_zero -> t.zero_over_zero <- t.zero_over_zero + 1
    | Unavailable Nonfinite_operands -> t.nonfinite <- t.nonfinite + 1
    | Unavailable Arithmetic_overflow -> t.overflow <- t.overflow + 1

  let finish t shape : Magnitude_summary.t =
    let has = t.finite > 0 in
    let scalar f : Difference_statistics.scalar =
      if not has then Unavailable Empty_finite_set
      else if Accum.Welford.overflowed t.welford then
        Unavailable Numeric_overflow
      else Finite_value (f ())
    in
    {
      finite_values = i64 t.finite;
      zero_over_zero = i64 t.zero_over_zero;
      unbounded = i64 t.unbounded;
      nonfinite_operands = i64 t.nonfinite;
      arithmetic_overflow = i64 t.overflow;
      minimum_finite = (if has then Some t.min else None);
      maximum_finite = (if has then Some t.max else None);
      largest_finite_index =
        (if has then Some (F.unravel shape t.max_flat) else None);
      mean_finite = scalar (fun () -> Accum.Welford.mean t.welford);
      standard_deviation_finite =
        (if not has then Unavailable Empty_finite_set
         else
           match Accum.Welford.deviation t.welford ~sample:false with
           | Some v when Float.is_finite v -> Finite_value v
           | _ -> Unavailable Numeric_overflow);
    }
end

(* ---- the comparison ------------------------------------------------------ *)

let max_ulp_update cur d =
  match cur with
  | None -> Some d
  | Some c -> if Int64.unsigned_compare d c > 0 then Some d else cur

let codec_of ctx v =
  match E.Registry.find ctx.L.Context.codecs (L.spec v).dtype.codec with
  | Some c -> Ok c
  | None ->
      fail (Codec (Codecs.Codec_error.Unknown_codec (L.spec v).dtype.codec))

let compare_integers ~max_samples ctx (policy : Comparison_policy.t) codec left
    right : Element_result.t =
  ignore ctx;
  let n = L.element_count left in
  let shape = (L.spec left).logical_shape in
  let lc = L.contents left and rc = L.contents right in
  let mismatches = ref 0 and first = ref None in
  let positive = ref 0 and negative = ref 0 and zero = ref 0 in
  let samples = ref [] in
  let signed = codec.E.element = Signed_integer in
  for i = 0 to n - 1 do
    let a = E.integer_value codec lc i and b = E.integer_value codec rc i in
    let c = if signed then Int64.compare a b else Int64.unsigned_compare a b in
    if c = 0 then incr zero
    else (
      incr mismatches;
      if !first = None then first := Some i;
      (* The residual sign follows the policy's direction. *)
      let right_larger = c < 0 in
      let positive_residual =
        match policy.statistics.direction with
        | Right_minus_left -> right_larger
        | Left_minus_right -> not right_larger
      in
      if positive_residual then incr positive else incr negative;
      if List.length !samples < max_samples then
        samples :=
          {
            Difference_measurement.index = F.unravel shape i;
            absolute = Unavailable Nonfinite_operands;
            relative = Unavailable Nonfinite_operands;
          }
          :: !samples)
  done;
  let unsupported : Difference_statistics.scalar =
    Unavailable Unsupported_dtype
  in
  let none_summary : Magnitude_summary.t =
    {
      finite_values = 0L;
      zero_over_zero = 0L;
      unbounded = 0L;
      nonfinite_operands = 0L;
      arithmetic_overflow = 0L;
      minimum_finite = None;
      maximum_finite = None;
      largest_finite_index = None;
      mean_finite = unsupported;
      standard_deviation_finite = unsupported;
    }
  in
  let rule = Option.is_some policy.element in
  {
    statistics =
      {
        status = Not_supported;
        finite_pairs = i64 n;
        excluded_nonfinite_pairs = 0L;
        positive_errors = i64 !positive;
        negative_errors = i64 !negative;
        zero_errors = i64 !zero;
        minimum_signed_error = None;
        maximum_signed_error = None;
        largest_absolute_error = None;
        bias = unsupported;
        standard_deviation = unsupported;
        mean_absolute_error = unsupported;
        root_mean_square_error = unsupported;
      };
    bitwise_mismatches = i64 !mismatches;
    tolerance_violations = (if rule then Some (i64 !mismatches) else None);
    first_bitwise_index = Option.map (F.unravel shape) !first;
    first_violation_index =
      (if rule then Option.map (F.unravel shape) !first else None);
    absolute = none_summary;
    relative = none_summary;
    nonfinite = Nonfinite_summary.zero;
    difference_samples = List.rev !samples;
    max_ulp_distance = None;
    finite_pairs = 0L;
    nan_pairs = 0L;
    infinity_pairs = 0L;
  }

let compare_floats ~max_samples (policy : Comparison_policy.t) codec left right
    : Element_result.t =
  let n = L.element_count left in
  let shape = (L.spec left).logical_shape in
  let lc = L.contents left and rc = L.contents right in
  let width = codec.E.byte_width in
  let rule : Float_policy.t option =
    match policy.element with Some (Floating p) -> Some p | _ -> None
  in
  (* Signed-zero handling for ULP ranks follows the element rule if present;
     without one, zeros are distinguished so the diagnostic never hides them. *)
  let distinct_zero =
    match rule with Some p -> p.signed_zero = Distinct | None -> true
  in
  let stats = Stats.create policy.statistics.direction in
  let abs_m = Mags.create () and rel_m = Mags.create () in
  let bitwise = ref 0 and violations = ref 0 in
  let first_bitwise = ref None and first_violation = ref None in
  let nf = ref Nonfinite_summary.zero in
  let finite_pairs = ref 0 and nan_pairs = ref 0 and inf_pairs = ref 0 in
  let max_ulp = ref None in
  let samples = ref [] in
  let note_class (k : F.klass) =
    let z = !nf in
    match k with
    | Finite -> ()
    | Nan_both_same ->
        incr nan_pairs;
        nf := { z with both_nan_same_bits = Int64.succ z.both_nan_same_bits }
    | Nan_both_different ->
        incr nan_pairs;
        nf :=
          {
            z with
            both_nan_different_bits = Int64.succ z.both_nan_different_bits;
          }
    | Nan_left ->
        incr nan_pairs;
        nf := { z with left_nan_only = Int64.succ z.left_nan_only }
    | Nan_right ->
        incr nan_pairs;
        nf := { z with right_nan_only = Int64.succ z.right_nan_only }
    | Inf_same ->
        incr inf_pairs;
        nf :=
          { z with same_sign_infinities = Int64.succ z.same_sign_infinities }
    | Inf_opposite ->
        incr inf_pairs;
        nf :=
          {
            z with
            opposite_sign_infinities = Int64.succ z.opposite_sign_infinities;
          }
    | Inf_left ->
        incr inf_pairs;
        nf :=
          {
            z with
            left_infinite_right_finite = Int64.succ z.left_infinite_right_finite;
          }
    | Inf_right ->
        incr inf_pairs;
        nf :=
          {
            z with
            right_infinite_left_finite = Int64.succ z.right_infinite_left_finite;
          }
  in
  let accepts (p : Float_policy.t) (k : F.klass) a b abits bbits =
    match k with
    | Nan_both_same ->
        (p.required = Tolerance
        && match p.nan with Reject -> false | Both_nan | Same_bits -> true)
        || (p.required = Bitwise && p.nan <> Reject)
    | Nan_both_different -> p.required = Tolerance && p.nan = Both_nan
    | Nan_left | Nan_right | Inf_opposite | Inf_left | Inf_right -> false
    | Inf_same -> true
    | Finite -> (
        match p.required with
        | Bitwise -> Int64.equal abits bbits
        | Tolerance ->
            let zero_sign_differs =
              a = 0.0 && b = 0.0 && not (Int64.equal abits bbits)
            in
            (not (zero_sign_differs && p.signed_zero = Distinct))
            && (a = b
               ||
               let r = F.round p.arithmetic in
               let atol = r p.atol and rtol = r p.rtol in
               let d = r (Float.abs (a -. b)) in
               let s = F.scale_of p.scale a b in
               d <= r (atol +. r (rtol *. r s))))
  in
  for i = 0 to n - 1 do
    let abits = E.bits codec lc i and bbits = E.bits codec rc i in
    let a = E.float_value codec lc i and b = E.float_value codec rc i in
    let bit_diff = not (Int64.equal abits bbits) in
    if bit_diff then (
      incr bitwise;
      if !first_bitwise = None then first_bitwise := Some i);
    let k = F.classify a b ~abits ~bbits in
    note_class k;
    let violated =
      match rule with
      | None -> false
      | Some p -> not (accepts p k a b abits bbits)
    in
    if violated then (
      incr violations;
      if !first_violation = None then first_violation := Some i);
    let abs_meas, rel_meas =
      match k with
      | Finite ->
          incr finite_pairs;
          Stats.add_finite stats i a b;
          if a = 0.0 && b = 0.0 && bit_diff then
            nf :=
              {
                !nf with
                signed_zero_differences = Int64.succ !nf.signed_zero_differences;
              };
          max_ulp :=
            max_ulp_update !max_ulp
              (F.ulp_distance ~width ~distinct_zero abits bbits);
          (F.absolute a b, F.relative policy.relative_scale a b)
      | _ ->
          stats.excluded <- stats.excluded + 1;
          let u : Difference_measurement.magnitude =
            Unavailable Nonfinite_operands
          in
          (u, u)
    in
    Mags.add abs_m i abs_meas;
    (* Absolute summaries have no 0/0 and no unbounded case by definition. *)
    Mags.add rel_m i rel_meas;
    let sampled = if rule = None then bit_diff else violated in
    if sampled && List.length !samples < max_samples then
      samples :=
        {
          Difference_measurement.index = F.unravel shape i;
          absolute = abs_meas;
          relative = rel_meas;
        }
        :: !samples
  done;
  {
    statistics = Stats.finish stats ~variance:policy.statistics.variance shape;
    bitwise_mismatches = i64 !bitwise;
    tolerance_violations =
      (if rule = None then None else Some (i64 !violations));
    first_bitwise_index = Option.map (F.unravel shape) !first_bitwise;
    first_violation_index = Option.map (F.unravel shape) !first_violation;
    absolute = Mags.finish abs_m shape;
    relative = Mags.finish rel_m shape;
    nonfinite = !nf;
    difference_samples = List.rev !samples;
    max_ulp_distance = Option.map Ulp_distance.of_unsigned_bits !max_ulp;
    finite_pairs = i64 !finite_pairs;
    nan_pairs = i64 !nan_pairs;
    infinity_pairs = i64 !inf_pairs;
  }

let compare ?(max_samples = 8) ctx (policy : Comparison_policy.t) ~left ~right =
  if not (Tensor_spec.equal (L.spec left) (L.spec right)) then
    fail Spec_mismatch
  else
    match codec_of ctx left with
    | Error e -> Error e
    | Ok codec -> (
        let fits =
          match (policy.element, codec.element) with
          | None, _ -> true
          | Some (Floating _), (Float_binary32 | Float_binary64) -> true
          | Some Exact_boolean, Boolean -> true
          | Some Exact_integer, (Signed_integer | Unsigned_integer) -> true
          | _ -> false
        in
        if not fits then fail Policy_codec_mismatch
        else
          match codec.element with
          | Float_binary32 | Float_binary64 ->
              Ok (compare_floats ~max_samples policy codec left right)
          | Boolean | Signed_integer | Unsigned_integer ->
              Ok (compare_integers ~max_samples ctx policy codec left right))

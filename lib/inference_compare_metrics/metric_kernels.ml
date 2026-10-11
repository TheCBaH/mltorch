open Inference_compare_schema
module O = Metric_outcome

(* Per-group score kernels. Operands arrive as binary64 arrays (binary32 values
   are exact in binary64). Every kernel first requires finite operands: a
   metric never inherits an element rule's NaN acceptance, and a nonfinite
   value is reported as such, not scored. *)

let finite a = Array.for_all Float.is_finite a
let both_finite a b = finite a && finite b

let guard a b f : O.t =
  if not (both_finite a b) then Nonfinite
  else if Array.length a = 0 then Undefined
  else f ()

let residual (direction : Difference_statistics.direction) a b i =
  match direction with
  | Right_minus_left -> b.(i) -. a.(i)
  | Left_minus_right -> a.(i) -. b.(i)

let score v : O.t = if Float.is_finite v then Score v else Numerical_failure

let mean_absolute_error a b =
  guard a b (fun () ->
      let s = Accum.Sum.create () in
      Array.iteri (fun i _ -> Accum.Sum.add s (Float.abs (b.(i) -. a.(i)))) a;
      if Accum.Sum.overflowed s then Numerical_failure
      else score (Accum.Sum.value s /. float_of_int (Array.length a)))

let root_mean_square_error a b =
  guard a b (fun () ->
      let q = Accum.Sum_squares.create () in
      let overflow = ref false in
      Array.iteri
        (fun i _ ->
          let e = b.(i) -. a.(i) in
          if Float.is_finite e then Accum.Sum_squares.add q e
          else overflow := true)
        a;
      if !overflow then Numerical_failure
      else score (Accum.Sum_squares.root_mean q (float_of_int (Array.length a))))

let welford direction a b =
  let w = Accum.Welford.create () in
  Array.iteri (fun i _ -> Accum.Welford.add w (residual direction a b i)) a;
  w

let absolute_bias direction a b =
  guard a b (fun () ->
      let w = welford direction a b in
      if Accum.Welford.overflowed w then Numerical_failure
      else score (Float.abs (Accum.Welford.mean w)))

let error_standard_deviation (p : Difference_statistics.policy) a b =
  guard a b (fun () ->
      let w = welford p.direction a b in
      if Accum.Welford.overflowed w then Numerical_failure
      else
        match Accum.Welford.deviation w ~sample:(p.variance = Sample) with
        | None -> Undefined
        | Some v -> score v)

let norm a =
  let q = Accum.Sum_squares.create () in
  Array.iter (Accum.Sum_squares.add q) a;
  Accum.Sum_squares.norm q

let normalized_l2 (scale : Float_policy.scale) (zero : Tensor_metric.zero_norm)
    a b =
  guard a b (fun () ->
      let r = Array.mapi (fun i x -> b.(i) -. x) a in
      if not (finite r) then Numerical_failure
      else
        let rn = norm r and an = norm a and bn = norm b in
        if not (Float.is_finite rn && Float.is_finite an && Float.is_finite bn)
        then Numerical_failure
        else
          let denom =
            match scale with
            | Left_scale -> an
            | Right_scale -> bn
            | Symmetric_max_scale -> Float.max an bn
          in
          if denom = 0.0 then
            if rn = 0.0 && zero = Both_zero_equal then Score 0.0 else Undefined
          else score (rn /. denom))

let max_abs a = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0.0 a

(* Roundoff tolerance beyond [-1, 1]: anything farther is a numerical failure
   under this implementation, not clamped. *)
let cosine_slack = 1e-8

let cosine_similarity (zero : Tensor_metric.zero_norm) a b =
  guard a b (fun () ->
      let ma = max_abs a and mb = max_abs b in
      if ma = 0.0 || mb = 0.0 then
        if ma = 0.0 && mb = 0.0 && zero = Both_zero_equal then Score 1.0
        else Undefined
      else
        let dot = Accum.Sum.create ()
        and na = Accum.Sum.create ()
        and nb = Accum.Sum.create () in
        Array.iteri
          (fun i _ ->
            let x = a.(i) /. ma and y = b.(i) /. mb in
            Accum.Sum.add dot (x *. y);
            Accum.Sum.add na (x *. x);
            Accum.Sum.add nb (y *. y))
          a;
        let sim =
          Accum.Sum.value dot
          /. (Float.sqrt (Accum.Sum.value na) *. Float.sqrt (Accum.Sum.value nb))
        in
        if not (Float.is_finite sim) then Numerical_failure
        else if Float.abs sim > 1.0 +. cosine_slack then Numerical_failure
        else Score (Float.max (-1.0) (Float.min 1.0 sim)))

let wasserstein_empirical a b =
  guard a b (fun () ->
      let sa = Array.copy a and sb = Array.copy b in
      Array.sort Float.compare sa;
      Array.sort Float.compare sb;
      let n = float_of_int (Array.length a) in
      let s = Accum.Sum.create () in
      let overflow = ref false in
      Array.iteri
        (fun i x ->
          let d = Float.abs (x -. sb.(i)) in
          if Float.is_finite d then Accum.Sum.add s (d /. n)
          else overflow := true)
        sa;
      if !overflow || Accum.Sum.overflowed s then Numerical_failure
      else score (Accum.Sum.value s))

let total a =
  let s = Accum.Sum.create () in
  Array.iter (Accum.Sum.add s) a;
  s

let wasserstein_histogram (mass : Tensor_metric.mass_policy)
    (support : float array) a b =
  guard a b (fun () ->
      let n = Array.length a in
      let increasing =
        Array.length support = n
        && Array.for_all Float.is_finite support
        &&
        let ok = ref true in
        for i = 0 to n - 2 do
          if not (support.(i) < support.(i + 1)) then ok := false
        done;
        !ok
      in
      if not increasing then Undefined
      else if
        Array.exists (fun x -> x < 0.0) a || Array.exists (fun x -> x < 0.0) b
      then Undefined
      else
        let ta = total a and tb = total b in
        if Accum.Sum.overflowed ta || Accum.Sum.overflowed tb then
          Numerical_failure
        else
          let ta = Accum.Sum.value ta and tb = Accum.Sum.value tb in
          if not (ta > 0.0 && tb > 0.0) then Undefined
          else
            let admissible =
              match mass with
              | Normalize_positive_total -> true
              | Require_unit_mass { absolute_tolerance } ->
                  Float.abs (ta -. 1.0) <= absolute_tolerance
                  && Float.abs (tb -. 1.0) <= absolute_tolerance
            in
            if not admissible then Undefined
            else
              (* Cumulative mass difference D_i; W1 = sum |D_i| (x_{i+1} - x_i). *)
              let cum = Accum.Sum.create () and w = Accum.Sum.create () in
              for i = 0 to n - 2 do
                Accum.Sum.add cum ((a.(i) /. ta) -. (b.(i) /. tb));
                Accum.Sum.add w
                  (Float.abs (Accum.Sum.value cum)
                  *. (support.(i + 1) -. support.(i)))
              done;
              if Accum.Sum.overflowed cum || Accum.Sum.overflowed w then
                Numerical_failure
              else score (Accum.Sum.value w))

(* The distribution a [Tensor_spec.Random] payload samples from. The parameters
   are spec configuration (human-written), so plain JSON numbers are fine — only
   the *sampled* values need f32-canonical treatment (see [Pcg]). *)

type t =
  | Normal of { mean : float; variance : float }
  | Uniform of { low : float; high : float }

let uniform_payload : t Jsont.t =
  Jsont.Object.map ~kind:"uniform" (fun low high -> Uniform { low; high })
  |> Jsont.Object.mem "low" Jsont.number
  |> Jsont.Object.mem "high" Jsont.number
  |> Jsont.Object.finish

let normal_payload : t Jsont.t =
  Jsont.Object.map ~kind:"normal" (fun mean variance ->
      Normal { mean; variance })
  |> Jsont.Object.mem "mean" Jsont.number
  |> Jsont.Object.mem "variance" Jsont.number
  |> Jsont.Object.finish

let jsont : t Jsont.t =
  Jsont.map ~kind:"distribution"
    ~dec:
      (Spec_util.union ~kind:"distribution"
         [
           ("normal", fun v -> Spec_util.dec normal_payload v);
           ("uniform", fun v -> Spec_util.dec uniform_payload v);
         ])
    ~enc:(function
      | Normal { mean; variance } ->
          Spec_util.single ~case:"normal"
            (Spec_util.jobj
               [
                 ("mean", Spec_util.jnum mean);
                 ("variance", Spec_util.jnum variance);
               ])
      | Uniform { low; high } ->
          Spec_util.single ~case:"uniform"
            (Spec_util.jobj
               [ ("low", Spec_util.jnum low); ("high", Spec_util.jnum high) ]))
    Jsont.json

open Inference_compare_schema
module Plan = Inference_compare_plan

let fail e = Policy_error.fail e

let validate structural (e : Experiment.t) =
  let correspondence = Plan.Structural.correspondence structural in
  let order = Plan.Structural.order structural in
  let finals = Plan.Structural.final_values structural in
  let rec go seen = function
    | [] -> Ok ()
    | (o : Donor_override.t) :: rest -> (
        let bad () = fail (Policy_error.make ~unit:o.unit Unknown_override) in
        let key =
          (Id.Unit.to_string o.unit, Id.Canonical_value.to_string o.output)
        in
        if List.mem key seen then bad ()
        else
          match Plan.Correspondence_check.writer correspondence o.output with
          | Some w when Id.Unit.equal w o.unit ->
              let used =
                Plan.Unit_order.consumers order o.output <> []
                || List.exists (Id.Canonical_value.equal o.output) finals
              in
              if used then go (key :: seen) rest else bad ()
          | _ -> bad ())
  in
  go [] e.overrides

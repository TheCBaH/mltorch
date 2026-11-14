type key = {
  target : string;
  numerics : string;
  candidate : string;
  baseline : string;
  version : string;
}

type t = { key : key; report : Paired_timing.report }
type reason = Capacity | Code_size

type kept =
  | Inconclusive_verdict
  | No_verdict
  | Output_differs
  | Regression of float

type choice =
  | Accepted of float
  | Candidate_for of reason * float option
  | Keep_baseline of kept

let find verdicts key = List.find_opt (fun v -> v.key = key) verdicts

let choose verdicts key ?reason () =
  match find verdicts key with
  | None -> Keep_baseline No_verdict
  | Some v -> (
      match v.report.Paired_timing.decision with
      | Paired_timing.Accept -> Accepted v.report.Paired_timing.median
      | Paired_timing.Output_differs -> Keep_baseline Output_differs
      | Paired_timing.Regression -> (
          match reason with
          | Some r -> Candidate_for (r, Some v.report.Paired_timing.median)
          | None -> Keep_baseline (Regression v.report.Paired_timing.median))
      | Paired_timing.Inconclusive -> (
          match reason with
          | Some r -> Candidate_for (r, Some v.report.Paired_timing.median)
          | None -> Keep_baseline Inconclusive_verdict))

let current verdicts ~version =
  List.filter (fun v -> String.equal v.key.version version) verdicts

let covered verdicts keys =
  match List.find_opt (fun k -> find verdicts k = None) keys with
  | None -> Ok ()
  | Some k -> Error k

let pp_choice ppf = function
  | Accepted g -> Fmt.pf ppf "candidate, accepted at %+.1f%%" g
  | Candidate_for (r, g) ->
      Fmt.pf ppf "candidate, for %s%a"
        (match r with Capacity -> "capacity" | Code_size -> "code size")
        (Fmt.option (fun ppf g -> Fmt.pf ppf " (speed %+.1f%%)" g))
        g
  | Keep_baseline No_verdict -> Fmt.string ppf "baseline kept: no verdict"
  | Keep_baseline Inconclusive_verdict ->
      Fmt.string ppf "baseline kept: inconclusive"
  | Keep_baseline Output_differs ->
      Fmt.string ppf "baseline kept: output differs"
  | Keep_baseline (Regression g) ->
      Fmt.pf ppf "baseline kept: regression at %+.1f%%" g

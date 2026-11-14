type pair = { old_time : float; new_time : float; output_matches : bool }
type decision = Accept | Inconclusive | Output_differs | Regression

type report = {
  gains : float list;
  median : float;
  low : float;
  high : float;
  decision : decision;
}

type error = Non_positive_time | Too_few_pairs | Too_many_pairs

let warmup = 3
let kept = 15
let gain ~old_time ~new_time = 100. *. log (old_time /. new_time)

let decide pairs =
  let n = List.length pairs in
  if n < warmup + kept then Error Too_few_pairs
  else if n > warmup + kept then Error Too_many_pairs
  else if
    List.exists
      (fun p ->
        (not (p.old_time > 0. && p.new_time > 0.))
        || Float.is_nan p.old_time || Float.is_nan p.new_time)
      pairs
  then Error Non_positive_time
  else
    let used = List.filteri (fun i _ -> i >= warmup) pairs in
    let gains =
      List.map (fun p -> gain ~old_time:p.old_time ~new_time:p.new_time) used
    in
    let sorted = Array.of_list (List.sort Float.compare gains) in
    let median = sorted.(7) and low = sorted.(3) and high = sorted.(11) in
    let decision =
      if List.exists (fun p -> not p.output_matches) pairs then Output_differs
      else if low > 0. then Accept
      else if high < 0. then Regression
      else Inconclusive
    in
    Ok { gains; median; low; high; decision }

let pp_decision ppf = function
  | Accept -> Fmt.string ppf "accept"
  | Inconclusive -> Fmt.string ppf "inconclusive"
  | Output_differs -> Fmt.string ppf "refused: output differs"
  | Regression -> Fmt.string ppf "regression"

let pp_report ppf r =
  Fmt.pf ppf "median %+.1f%%, interval [%+.1f, %+.1f]: %a" r.median r.low r.high
    pp_decision r.decision

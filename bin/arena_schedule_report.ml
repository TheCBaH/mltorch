(* Markdown summary of the scheduling evaluation's rows. Pool bytes are the
   figure that counts; payload peaks and the unchanged/regressed counts keep a
   payload win that fragments from passing as a saving. *)

module Plan = Arena_schedule_plan
open Core.Storage_units

type row = {
  model : string;
  setting : string;
  nodes : int;
  selection : Plan.Selection.t;
}

let pool (b : Byte_size.t option) =
  match b with Some b -> Byte_size.to_int64 b | None -> 0L

(* [None] on a zero denominator. *)
let percent ~before ~after =
  if Int64.equal before 0L then None
  else
    Some
      (100. *. Int64.to_float (Int64.sub before after) /. Int64.to_float before)

let pp_percent ppf = function
  | None -> Fmt.string ppf "n/a"
  | Some p -> Fmt.pf ppf "%.2f%%" p

let settings rows =
  List.fold_left
    (fun acc r -> if List.mem r.setting acc then acc else acc @ [ r.setting ])
    [] rows

let render ppf rows =
  Fmt.pf ppf "# Memory scheduling evaluation@.@.";
  Fmt.pf ppf
    "Pool bytes are the allocated bytes of the production planner's witnessed \
     placement; the original order is always a candidate. No timings, no \
     weights, no RSS claim.@.@.";
  List.iter
    (fun setting ->
      let rs = List.filter (fun r -> r.setting = setting) rows in
      let count p = List.length (List.filter p rs) in
      let base r = pool r.selection.Plan.Selection.baseline_pool_bytes
      and chosen r = pool r.selection.pool_bytes in
      let sum f = List.fold_left (fun a r -> Int64.add a (f r)) 0L rs in
      let reduced = count (fun r -> chosen r < base r)
      and regressed = count (fun r -> chosen r > base r)
      and unchanged = count (fun r -> chosen r = base r) in
      let target r = Byte_size.to_int64 r.selection.metrics.target_peak
      and target0 r = Byte_size.to_int64 r.selection.baseline.target_peak in
      let disagree =
        count (fun r ->
            let winner =
              List.find_opt
                (fun (c : Plan.Report.t) ->
                  c.strategy = r.selection.payload_winner)
                r.selection.reports
            in
            match winner with
            | Some w -> pool w.pool_bytes > chosen r
            | None -> false)
      in
      Fmt.pf ppf "## %s@.@." setting;
      Fmt.pf ppf
        "- models: %d; pool reduced: %d, unchanged: %d, regressed: %d@."
        (List.length rs) reduced unchanged regressed;
      Fmt.pf ppf "- payload winner would have grown the pool: %d@." disagree;
      Fmt.pf ppf "- pool bytes, paired total: %Ld -> %Ld (%a)@." (sum base)
        (sum chosen) pp_percent
        (percent ~before:(sum base) ~after:(sum chosen));
      Fmt.pf ppf "- target payload peak, paired total: %Ld -> %Ld (%a)@.@."
        (sum target0) (sum target) pp_percent
        (percent ~before:(sum target0) ~after:(sum target));
      let top =
        List.filter (fun r -> chosen r < base r) rs
        |> List.sort (fun a b ->
            Int64.compare
              (Int64.sub (base b) (chosen b))
              (Int64.sub (base a) (chosen a)))
        |> List.filteri (fun i _ -> i < 15)
      in
      if top <> [] then (
        Fmt.pf ppf
          "| model | nodes | pool before | pool after | saved | strategy |@.";
        Fmt.pf ppf "| --- | ---: | ---: | ---: | ---: | --- |@.";
        List.iter
          (fun r ->
            Fmt.pf ppf "| %s | %d | %Ld | %Ld | %a | %a |@." r.model r.nodes
              (base r) (chosen r) pp_percent
              (percent ~before:(base r) ~after:(chosen r))
              Arena_schedule.Strategy.pp r.selection.strategy)
          top;
        Fmt.pf ppf "@."))
    (settings rows)

(* Every model once: its original pool bytes, then the chosen pool bytes under
   each configuration, in corpus order. *)
let render_all ppf rows =
  let names = settings rows in
  let models =
    List.fold_left
      (fun acc r -> if List.mem r.model acc then acc else acc @ [ r.model ])
      [] rows
  in
  Fmt.pf ppf "## All models (pool bytes)@.@.";
  Fmt.pf ppf "| model | nodes | original | %s |@." (String.concat " | " names);
  Fmt.pf ppf "| --- | ---: | ---: |%s@."
    (String.concat "" (List.map (fun _ -> " ---: |") names));
  List.iter
    (fun model ->
      let of_model = List.filter (fun r -> r.model = model) rows in
      let first = List.hd of_model in
      Fmt.pf ppf "| %s | %d | %Ld |" model first.nodes
        (pool first.selection.baseline_pool_bytes);
      List.iter
        (fun name ->
          match List.find_opt (fun r -> r.setting = name) of_model with
          | Some r -> Fmt.pf ppf " %Ld |" (pool r.selection.pool_bytes)
          | None -> Fmt.pf ppf " - |")
        names;
      Fmt.pf ppf "@.")
    models;
  Fmt.pf ppf "@."

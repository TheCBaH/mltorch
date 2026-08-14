(* See native_predict.mli for the contract. Two decisions carry the weight here,
   and both are load-bearing rather than stylistic:

   1. Ranking reads the LOGITS. Ranking the probabilities instead is correct in
      exact arithmetic and wrong in floating point: [exp (l -. max)] underflows
      to 0. well before the logit gap stops being representable, so two classes
      whose logits differ can end up with identical 0. probabilities and be
      ordered by the tie-break instead of by the model. Ranking the logits keeps
      the order the model actually produced.
   2. The softmax denominator spans EVERY class. Normalizing over the selected
      [k] would make the printed probabilities sum to 1 across the top five and
      disagree with the ATen runner, whose [_softmax] runs over the whole class
      axis before [topk]. *)

module Too_few_classes = struct
  type t = { classes : int; wanted : int }
end

module Non_finite = struct
  type t = { index : int; value : float }
end

type error =
  [ `Invalid_k of int
  | `Output_count of int
  | `Not_class_logits of Vec6.shape
  | `Too_few_classes of Too_few_classes.t
  | `Non_finite_logit of Non_finite.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Invalid_k k -> Fmt.pf ppf "top-k needs k >= 1, got %d" k
  | `Output_count n -> Fmt.pf ppf "expected exactly one output tensor, got %d" n
  | `Not_class_logits shape ->
      Fmt.pf ppf "not one batch of class logits: %a" Vec6.pp_shape shape
  | `Too_few_classes { Too_few_classes.classes; wanted } ->
      Fmt.pf ppf "top-%d requested, only %d class%s" wanted classes
        (if classes = 1 then "" else "es")
  | `Non_finite_logit { Non_finite.index; value } ->
      Fmt.pf ppf "non-finite logit at class %d: %h" index value

(* Every axis but [C] must be a single element: one batch of class logits. *)
let batch_axes = [ Axis.N; Axis.T; Axis.D; Axis.H; Axis.W ]

let logits_of_output (Tensor.Tensor t as packed) =
  let shape = t.Tensor.shape in
  let extent axis = Dim.to_int (Vec6.get shape axis) in
  if not (List.for_all (fun a -> extent a = 1) batch_axes) then
    Error (`Not_class_logits shape)
  else
    Ok
      (Array.init (extent Axis.C) (fun c ->
           Tensor.read packed (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c)))

(* [None] when every logit is finite. Reports the first offender in index order
   so the failure is the same on both backends. *)
let first_non_finite logits =
  let rec go i =
    if i >= Array.length logits then None
    else if Float.is_finite logits.(i) then go (i + 1)
    else Some { Non_finite.index = i; value = logits.(i) }
  in
  go 0

(* Total: descending by logit, then ascending by class index. [List.sort] is not
   documented stable, so the tie-break has to be in the comparator rather than
   left to the sort. *)
let by_rank (ia, la) (ib, lb) =
  match Float.compare lb la with 0 -> Int.compare ia ib | c -> c

let top_predictions outputs k =
  let open Err.Syntax in
  let* () = if k >= 1 then Err.return () else Err.fail (`Invalid_k k) in
  let* logits =
    match outputs with
    | [ output ] -> (
        match logits_of_output output with
        | Ok logits -> Err.return logits
        | Error e -> Err.fail e)
    | outputs -> Err.fail (`Output_count (List.length outputs))
  in
  let classes = Array.length logits in
  let* () =
    if classes >= k then Err.return ()
    else Err.fail (`Too_few_classes { Too_few_classes.classes; wanted = k })
  in
  let* () =
    match first_non_finite logits with
    | None -> Err.return ()
    | Some payload -> Err.fail (`Non_finite_logit payload)
  in
  let ranked =
    Array.to_list (Array.mapi (fun i l -> (i, l)) logits) |> List.sort by_rank
  in
  (* The stable softmax, over all [classes] logits. *)
  let max_logit = Array.fold_left Float.max logits.(0) logits in
  let denominator =
    Array.fold_left (fun acc l -> acc +. Float.exp (l -. max_logit)) 0. logits
  in
  Err.return
    (List.filteri (fun i _ -> i < k) ranked
    |> List.map (fun (i, l) -> (i, Float.exp (l -. max_logit) /. denominator)))

(* One side of a comparison: a graph view of the model and the configuration of
   the engine that runs it. The views are the exporter graph, the lowered
   Native graph, the Native graph after the canonical passes, and the
   four-axis graph; the configuration is how the Native direct evaluator
   accumulates dot products. *)

module View = struct
  type t = Aten | Native | Native4d | Native_canonical

  (* The order the views are derived from one another, which is the order
     their correspondences compose in. *)
  let rank = function
    | Aten -> 0
    | Native -> 1
    | Native_canonical -> 2
    | Native4d -> 3

  let to_string = function
    | Aten -> "aten"
    | Native -> "native"
    | Native4d -> "native4d"
    | Native_canonical -> "native.canonical"

  let of_string = function
    | "aten" -> Some Aten
    | "native" -> Some Native
    | "native.canonical" -> Some Native_canonical
    | "native4d" -> Some Native4d
    | _ -> None
end

(* How the Native-family views are evaluated: concretely, or by grounding the
   whole-graph symbolic form. *)
module Evaluator = struct
  (* [Provider name] is an engine registered by a library above this one. *)
  type t = Direct | Provider of string | Symbolic
end

type t = {
  view : View.t;
  evaluator : Evaluator.t;
  dots : Direct.dot_accumulation;
  storage : Storage_config.t;
}

let make ?(evaluator = Evaluator.Direct) ?(dots = Direct.Binary64)
    ?(storage = Storage_config.default) view =
  { view; evaluator; dots; storage }

let sequential32 = "sequential32"
let symbolic = "symbolic"

(* The numerics words of the compiled engines. An engine registers one provider
   for each numerics it runs, under the engine word and the word as a suffix:
   [loop-c,fp32] is the provider [loop-c-fp32]. [fp32] is the engine's binary32
   configuration as deployed (vectors where it has them); [fp32s] the same
   policy kept scalar, for an engine that has both. *)
let fp32 = "fp32"
let fp32s = "fp32s"
let numerics_words = [ fp32; fp32s ]

let is_fp32_engine w =
  List.exists
    (fun word ->
      let suffix = "-" ^ word in
      let n = String.length w and k = String.length suffix in
      n > k && String.equal (String.sub w (n - k) k) suffix)
    numerics_words

let ( let* ) = Result.bind

(* [VIEW], or [VIEW:WORD,WORD...]: [sequential32] for binary32 sequential dots,
   the storage words of {!Storage_config}, or [symbolic] for the symbolic
   evaluator, which takes no other word; [native-sequential] is the earlier
   spelling of [native:sequential32]. Only the Native-family views take
   configuration, and Native4D only [symbolic]. *)
let of_string s =
  let view, words =
    match s with
    | "native-sequential" -> ("native", [ sequential32 ])
    | _ -> (
        match String.index_opt s ':' with
        | None -> (s, [])
        | Some i ->
            ( String.sub s 0 i,
              String.split_on_char ','
                (String.sub s (i + 1) (String.length s - i - 1)) ))
  in
  match View.of_string view with
  | None -> Error (Printf.sprintf "unknown variant %S" s)
  | Some view ->
      let native =
        match view with View.Native | Native_canonical -> true | _ -> false
      in
      let rec go dots storage = function
        | [] -> Ok (dots, List.rev storage)
        | w :: rest when native && String.equal w sequential32 ->
            go Direct.Binary32_sequential storage rest
        | w :: rest -> (
            match (native, Storage_config.word_of_string w) with
            | true, Some word -> go dots (word :: storage) rest
            | _ -> Error (Printf.sprintf "variant %S: no configuration %S" s w))
      in
      let* words, numerics_requested =
        match List.find_opt (fun w -> List.mem w numerics_words) words with
        | None -> Ok (words, None)
        | Some word ->
            if
              (View.rank view = 1 || View.rank view = 2)
              && List.exists (fun w -> Provider.find w <> None) words
            then Ok (List.filter (fun w -> w <> word) words, Some word)
            else
              Error
                (Printf.sprintf "variant %S: %s needs an engine word" s word)
      in
      let* words =
        match numerics_requested with
        | None -> Ok words
        | Some word -> (
            match List.find_opt (fun w -> Provider.find w <> None) words with
            | None -> Ok words
            | Some w ->
                let w32 = w ^ "-" ^ word in
                if Provider.find w32 <> None then
                  Ok (List.map (fun x -> if x = w then w32 else x) words)
                else
                  Error
                    (Printf.sprintf "variant %S: engine %S has no %s numerics" s
                       w word))
      in
      let evaluator, words =
        let provided =
          if View.rank view = 1 || View.rank view = 2 then
            List.find_opt (fun w -> Provider.find w <> None) words
          else None
        in
        match provided with
        | Some w -> (Evaluator.Provider w, List.filter (fun x -> x <> w) words)
        | None ->
            if View.rank view > 0 && List.mem symbolic words then
              (Evaluator.Symbolic, List.filter (fun w -> w <> symbolic) words)
            else (Evaluator.Direct, words)
      in
      let* dots, words = go Direct.Binary64 [] words in
      let* storage =
        Storage_config.of_words words
        |> Result.map_error (Printf.sprintf "variant %S: %s" s)
      in
      if
        evaluator <> Evaluator.Direct
        && (dots <> Direct.Binary64 || not (Storage_config.is_default storage))
      then
        Error
          (Printf.sprintf "variant %S: %s takes no other word" s
             (match evaluator with Evaluator.Provider w -> w | _ -> symbolic))
      else Ok (make ~evaluator ~dots ~storage view)

(* Whether the configuration runs every kernel in binary32 under the
   [Fp32_ordered] policy: two such variants are bitwise equal by the policy's
   contract, so a pair of them is compared exactly unless asked otherwise. *)
let is_fp32 t =
  match t.evaluator with
  | Evaluator.Provider w -> is_fp32_engine w
  | Evaluator.Direct | Evaluator.Symbolic -> false

let to_string t =
  let words =
    (match t.evaluator with
      | Evaluator.Direct -> []
      | Evaluator.Provider w -> [ w ]
      | Evaluator.Symbolic -> [ symbolic ])
    @ (match t.dots with
      | Direct.Binary64 -> []
      | Direct.Binary32_sequential -> [ sequential32 ])
    @ Storage_config.to_words t.storage
  in
  match words with
  | [] -> View.to_string t.view
  | words -> View.to_string t.view ^ ":" ^ String.concat "," words

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

type t = {
  view : View.t;
  dots : Direct.dot_accumulation;
  storage : Storage_config.t;
}

let make ?(dots = Direct.Binary64) ?(storage = Storage_config.default) view =
  { view; dots; storage }

let sequential32 = "sequential32"
let ( let* ) = Result.bind

(* [VIEW], or [VIEW:WORD,WORD...]: [sequential32] for binary32 sequential dots
   and the storage words of {!Storage_config}; [native-sequential] is the
   earlier spelling of [native:sequential32]. Only the Native views take
   configuration: the others have none to give. *)
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
      let* dots, words = go Direct.Binary64 [] words in
      let* storage =
        Storage_config.of_words words
        |> Result.map_error (Printf.sprintf "variant %S: %s" s)
      in
      Ok (make ~dots ~storage view)

let to_string t =
  let words =
    (match t.dots with
      | Direct.Binary64 -> []
      | Direct.Binary32_sequential -> [ sequential32 ])
    @ Storage_config.to_words t.storage
  in
  match words with
  | [] -> View.to_string t.view
  | words -> View.to_string t.view ^ ":" ^ String.concat "," words

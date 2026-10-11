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

type t = { view : View.t; dots : Direct.dot_accumulation }

let make ?(dots = Direct.Binary64) view = { view; dots }
let sequential32 = "sequential32"

(* [VIEW] or [VIEW:sequential32]; [native-sequential] is the earlier spelling
   of [native:sequential32]. Only the Native views accumulate dot products. *)
let of_string s =
  let view, config =
    match s with
    | "native-sequential" -> ("native", Some sequential32)
    | _ -> (
        match String.index_opt s ':' with
        | None -> (s, None)
        | Some i ->
            ( String.sub s 0 i,
              Some (String.sub s (i + 1) (String.length s - i - 1)) ))
  in
  match (View.of_string view, config) with
  | None, _ -> Error (Printf.sprintf "unknown variant %S" s)
  | Some view, None -> Ok (make view)
  | Some ((Native | Native_canonical) as view), Some c
    when String.equal c sequential32 ->
      Ok (make ~dots:Direct.Binary32_sequential view)
  | Some _, Some c ->
      Error (Printf.sprintf "variant %S: no configuration %S" s c)

let to_string t =
  match t.dots with
  | Direct.Binary64 -> View.to_string t.view
  | Direct.Binary32_sequential -> View.to_string t.view ^ ":" ^ sequential32

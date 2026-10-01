(* Generic-JSON builders and readers for the arena evaluation's JSONL. An
   [int64] quantity is a decimal string, so no consumer rounds it through a
   double. *)

module J = Jsont.Json

let obj members =
  J.object' (List.map (fun (k, v) -> J.mem (J.name k) v) members)

let str = J.string
let i64 = J.int64_as_string
let int = J.int
let num f = if Float.is_finite f then J.number f else J.null ()
let bool = J.bool
let null = J.null ()
let opt f = function Some v -> f v | None -> null
let list f xs = J.list (List.map f xs)

let encode json =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json json with
  | Ok line -> line
  | Error msg -> failwith msg

let decode line = Jsont_bytesrw.decode_string Jsont.json line

let field name = function
  | Jsont.Object (members, _) -> (
      match J.find_mem name members with Some (_, v) -> Some v | None -> None)
  | _ -> None

let to_string = function Jsont.String (s, _) -> Some s | _ -> None
let to_float = function Jsont.Number (f, _) -> Some f | _ -> None
let to_int j = Option.map Float.to_int (to_float j)
let to_bool = function Jsont.Bool (b, _) -> Some b | _ -> None
let to_list = function Jsont.Array (xs, _) -> Some xs | _ -> None
let to_int64 j = Option.bind (to_string j) Int64.of_string_opt
let get conv name j = Option.bind (field name j) conv
let string_field name j = get to_string name j

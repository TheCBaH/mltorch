(* Test-only JSON observations. Every value printed is read from the CLI result. *)
include Transformers_metadata.Json_util
module J = Jsont.Json

let get value =
  Err.or_raise ~pp_error:Transformers_metadata.Fault.pp_error value

let at key value = get (member key value)
let str value = get (string value)
let field key value = str (at key value)
let arr value = get (array value)
let rows key value = arr (at key value)

let opt key value =
  match List.assoc_opt key (get (members value)) with
  | Some v -> v
  | None -> J.null ()

let optional_rows key value =
  match opt key value with Jsont.Null _ -> [] | v -> arr v

let read path = get (read path)
let boolean v = get (bool v)
let py_bool v = if v then "True" else "False"
let quote s = "'" ^ String.concat "\\'" (String.split_on_char '\'' s) ^ "'"

let rec repr = function
  | Jsont.Null _ -> "None"
  | Jsont.Bool (b, _) -> py_bool b
  | Jsont.String (s, _) -> quote s
  | Jsont.Number (n, _) -> Printf.sprintf "%.16g" n
  | Jsont.Array (xs, _) -> "[" ^ String.concat ", " (List.map repr xs) ^ "]"
  | Jsont.Object (xs, _) ->
      "{"
      ^ String.concat ", "
          (List.map (fun ((k, _), v) -> quote k ^ ": " ^ repr v) xs)
      ^ "}"

let tuple xs = "(" ^ String.concat ", " xs ^ ")"
let list xs = "[" ^ String.concat ", " xs ^ "]"

let graph s id =
  get
    (Err.of_option (`Metadata_missing id)
       (List.find_opt
          (fun g -> field "id" g = id)
          (List.concat_map (rows "graphs") (rows "graphCollections" s))))

let by key id xs =
  get
    (Err.of_option (`Metadata_missing id)
       (List.find_opt (fun x -> field key x = id) xs))

let cap s id = at "status" (by "key" id (rows "capabilities" s))
let nodes s id = rows "nodes" (graph s id)

let attrs x =
  List.map (fun a -> (field "key" a, at "value" a)) (optional_rows "attrs" x)

let attr key x = List.assoc_opt key (attrs x)

let operations s id =
  List.filter
    (fun n ->
      not (List.mem (field "label" n) [ "input"; "constant"; "output" ]))
    (nodes s id)

let contains text word =
  let rec loop i =
    i + String.length word <= String.length text
    && (String.sub text i (String.length word) = word || loop (i + 1))
  in
  loop 0

let primary x = not (String.starts_with ~prefix:"expr/" (field "id" x))
let ids xs = list (List.map (fun x -> quote (field "id" x)) xs)

let rec canonical = function
  | Jsont.Object (xs, _) ->
      obj
        (List.sort compare
           (List.map (fun ((key, _), v) -> (key, canonical v)) xs))
  | Jsont.Array (xs, _) -> J.list (List.map canonical xs)
  | v -> v

let equal a b = get (text (canonical a)) = get (text (canonical b))

let update key value x =
  obj ((key, value) :: List.remove_assoc key (get (members x)))

let map_nodes fn s =
  update "graphCollections"
    (J.list
       (List.map
          (fun c ->
            update "graphs"
              (J.list
                 (List.map
                    (fun g ->
                      update "nodes" (J.list (List.map fn (rows "nodes" g))) g)
                    (rows "graphs" c)))
              c)
          (rows "graphCollections" s)))
    s

let count_by labels =
  List.sort_uniq String.compare labels
  |> List.map (fun label ->
      (label, List.length (List.filter (( = ) label) labels)))

let counts xs =
  list (List.map (fun (name, n) -> tuple [ quote name; string_of_int n ]) xs)

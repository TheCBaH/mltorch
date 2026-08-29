type t =
  | Bool
  | Dict of t (* Dict[str, T] — key is always str *)
  | Float
  | Int
  | List of t
  | Optional of t
  | Ref of string
  | Str

let rec to_string = function
  | Bool -> "bool"
  | Dict t -> "Dict[str, " ^ to_string t ^ "]"
  | Float -> "float"
  | Int -> "int"
  | List t -> "List[" ^ to_string t ^ "]"
  | Optional t -> "Optional[" ^ to_string t ^ "]"
  | Ref s -> s
  | Str -> "str"

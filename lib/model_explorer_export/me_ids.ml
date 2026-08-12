(* Deterministic external identifiers. See the .mli. *)

type error =
  [ `Control_byte of int | `Label_too_long of int | `Id_too_long of int ]

let pp_error fmt : [< error ] -> unit = function
  | `Control_byte b -> Fmt.pf fmt "control byte 0x%02X in label" b
  | `Label_too_long n -> Fmt.pf fmt "label of %d bytes is over the ceiling" n
  | `Id_too_long n -> Fmt.pf fmt "encoded id of %d bytes is over the ceiling" n

(* The characters the renderer's own grammar reads, plus [%] itself. Encoded in
   ONE pass, which is what makes the map injective: a two-pass scheme that
   escaped [%] after the others would turn ["%2F"] and ["/"] into the same
   output. Their order in this list is irrelevant for the same reason. *)
let reserved = [ '%'; '/'; ';'; '\\'; '#'; ':' ]
let is_control c = Char.code c < 0x20 || Char.code c = 0x7F

let component s =
  let buf = Buffer.create (String.length s) in
  let rec go i =
    if i >= String.length s then Err.return (Buffer.contents buf)
    else
      let c = s.[i] in
      if is_control c then Err.fail (`Control_byte (Char.code c))
      else begin
        if List.mem c reserved then
          Buffer.add_string buf (Printf.sprintf "%%%02X" (Char.code c))
        else Buffer.add_char buf c;
        go (i + 1)
      end
  in
  go 0

module Layer = struct
  type t = Pt2 | Native | Native4d | Symbolic | Kernel

  let to_string = function
    | Pt2 -> "pt2"
    | Native -> "native"
    | Native4d -> "native4d"
    | Symbolic -> "symbolic"
    | Kernel -> "kernel"

  (* The same successor chain [Me_limits.Diagnostic.Code] uses, and for the
     same reason: a list written beside the type compiles while everything that
     iterates the vocabulary quietly stops seeing the new member. *)
  let next = function
    | Pt2 -> Some Native
    | Native -> Some Native4d
    | Native4d -> Some Symbolic
    | Symbolic -> Some Kernel
    | Kernel -> None

  let all =
    let rec walk acc c =
      match next c with
      | None -> List.rev (c :: acc)
      | Some n -> walk (c :: acc) n
    in
    walk [] Pt2

  let of_string s = List.assoc_opt s (List.map (fun l -> (to_string l, l)) all)
end

(* The two ceilings, applied where each belongs: the raw input before encoding,
   because that is the figure a caller can reason about, and the encoded result
   after, because that is what the renderer sees and encoding expands. *)

let check_label ~limits s =
  let n = String.length s in
  if n > limits.Me_limits.Limits.max_label_bytes then
    Err.fail (`Label_too_long n)
  else Err.return ()

let check_id ~max s =
  let n = String.length s in
  if n > max then Err.fail (`Id_too_long n) else Err.return s

let encoded_component ~limits s =
  let open Err.Syntax in
  let* () = check_label ~limits s in
  component s

(* Structural forms. Zero padding is for ordering in a rendered list, not a
   bound — the count is bounded by [max_graphs] and [max_states], and a value
   past 999 simply renders wider. *)
let ordinal = Printf.sprintf "%03d"

let collection ~limits name =
  let open Err.Syntax in
  let* encoded = encoded_component ~limits name in
  check_id ~max:limits.Me_limits.Limits.max_id_bytes ("mltorch:" ^ encoded)

let pt2_graph path =
  Core.Pretty.to_string
    (fun fmt p -> Fmt.pf fmt "pt2/%a" Pt2_native_graph.Graph_path.pp p)
    path

let pt2_node path index =
  Core.Pretty.to_string
    (fun fmt (p, i) -> Fmt.pf fmt "%a#%d" Pt2_native_graph.Graph_path.pp p i)
    (path, index)

let pt2_boundary ~limits kind name =
  let open Err.Syntax in
  let prefix =
    match kind with `In -> "in" | `Const -> "const" | `Out -> "out"
  in
  (* The SSA name is a dynamic component and goes through the encoder; the [:]
     that separates it from the prefix does not, being structural. *)
  let* encoded = encoded_component ~limits name in
  check_id ~max:limits.Me_limits.Limits.max_id_bytes
    (Printf.sprintf "%s:%s" prefix encoded)

let pt2_namespace ~limits levels =
  let open Err.Syntax in
  let* components =
    Err.List.map
      (fun l ->
        let* encoded = encoded_component ~limits l in
        check_id ~max:limits.Me_limits.Limits.max_namespace_component_bytes
          encoded)
      levels
  in
  (* Each LEVEL meets the per-component ceiling, because that is the unit the
     renderer's own splitting works over; the joined string meets
     [max_id_bytes], because that is what every node carries. *)
  check_id ~max:limits.Me_limits.Limits.max_id_bytes
    (String.concat "/" components)

let layered prefix layer n =
  Printf.sprintf "%s/%s/%s" prefix (Layer.to_string layer) (ordinal n)

let graph = layered "g"
let flow_state = layered "s"
let flow_transition = layered "t"
let op_node id = Printf.sprintf "n%d" (Graph_ir.Node_id.to_int id)

(* A VALUE graph's node, named by the value it produces. Deliberately a
   different letter from [op_node]: those graphs are node-indexed and these are
   value-indexed, and a stage and the kernel value adapted from it share this id
   precisely so the two can be compared without an explicit mapping. *)
let value_node id = Printf.sprintf "v%d" (Graph_ir.Tensor_id.to_int id)

let boundary kind id =
  let prefix =
    match kind with `In -> "in" | `Const -> "const" | `Out -> "out"
  in
  Printf.sprintf "%s:t%d" prefix (Graph_ir.Tensor_id.to_int id)

let namespace_component ~limits ?label id =
  let open Err.Syntax in
  let* assembled =
    match label with
    | None -> Err.return (Printf.sprintf "g%d" id)
    | Some l ->
        let+ encoded = encoded_component ~limits l in
        Printf.sprintf "%s#g%d" encoded id
  in
  check_id ~max:limits.Me_limits.Limits.max_namespace_component_bytes assembled

let detail ~limits ~parent_graph ~parent_node k =
  (* The parents are ids this module already produced. Re-encoding them would
     not be idempotent — the [%] of an existing escape would escape again — and
     the result would no longer name the graph it is derived from. *)
  check_id ~max:limits.Me_limits.Limits.max_id_bytes
    (Printf.sprintf "expr/%s/%s/t%d" parent_graph parent_node k)

open Ssa_ir

(* The emission state of one kernel: value names, the declarations function scope
   must hold, the runtime helpers and failure sites used, and the output. Every
   value is a C variable declared at function scope and assigned where it is
   defined, so a loop's carried values and an [if]'s results are plain
   assignments and nothing depends on C block scope. *)

module R = Loop_ir.Loop_c_runtime

type error =
  [ `Unsupported_format of Ssa_id.Buffer.t * string | `Unsupported_lanes of int ]

let pp_error ppf : [< error ] -> unit = function
  | `Unsupported_format (b, f) ->
      Fmt.pf ppf "%a: format %s has no C implementation" Ssa_id.Buffer.pp b f
  | `Unsupported_lanes n ->
      Fmt.pf ppf "a %d-lane vector is not a power of two" n

exception Refused of error

type t = {
  names : (int, string) Hashtbl.t;
  mutable decls : (string * string * string) list;  (** name, type, init *)
  mutable next_name : int;
  mutable next_temp : int;
  mutable used : R.Name.t list;
  mutable sites : Loop_ir.Loop_failure.t list;
  mutable site_count : int;
  mutable local_doubles : int64;
  mutable vector_types : (string * string) list;  (** typedef name, text *)
  mutable vector_helpers : (string * string) list;  (** function name, text *)
  mutable f32 : bool;
  locals : (int, int64 * Expr.Local_var.t option) Hashtbl.t;
      (** a scratch object's slots and the variable it names *)
  out : Buffer.t;
  buffers : Ssa_buffer.t list;
  site_table : Loop_ir.Loop_failure.t array option;
}

let create ?site_table buffers =
  {
    names = Hashtbl.create 64;
    decls = [];
    next_name = 0;
    next_temp = 0;
    used = [];
    sites = [];
    site_count = 0;
    local_doubles = 0L;
    vector_types = [];
    vector_helpers = [];
    f32 = false;
    locals = Hashtbl.create 8;
    out = Buffer.create 4096;
    buffers;
    site_table;
  }

let use cx h = if not (List.mem h cx.used) then cx.used <- h :: cx.used

let call cx h args =
  use cx h;
  R.Name.to_string h ^ "(" ^ String.concat ", " args ^ ")"

let line cx depth fmt =
  Printf.ksprintf
    (fun s ->
      Buffer.add_string cx.out (String.make (2 * depth) ' ');
      Buffer.add_string cx.out s;
      Buffer.add_char cx.out '\n')
    fmt

(* A power-of-two lane count, the one a generic vector type holds. *)
let check_lanes lanes =
  let n = Ssa_type.Lanes.to_int lanes in
  if n < 2 || n land (n - 1) <> 0 then raise (Refused (`Unsupported_lanes n));
  n

let vector_type cx ~elem lanes =
  let n = check_lanes lanes in
  let name, text =
    match elem with
    | `F32 ->
        ( Printf.sprintf "ssa_vs%d" n,
          Printf.sprintf
            "typedef float ssa_vs%d __attribute__((vector_size(%d)));" n (4 * n)
        )
    | `F64 ->
        ( Printf.sprintf "ssa_vd%d" n,
          Printf.sprintf
            "typedef double ssa_vd%d __attribute__((vector_size(%d)));" n (8 * n)
        )
    | `Mask ->
        ( Printf.sprintf "ssa_md%d" n,
          Printf.sprintf
            "typedef int64_t ssa_md%d __attribute__((vector_size(%d)));" n
            (8 * n) )
    | `Mask32 ->
        ( Printf.sprintf "ssa_mi%d" n,
          Printf.sprintf
            "typedef int32_t ssa_mi%d __attribute__((vector_size(%d)));" n
            (4 * n) )
  in
  if not (List.mem_assoc name cx.vector_types) then
    cx.vector_types <- cx.vector_types @ [ (name, text) ];
  name

(* The C type a value of this type is held in, or [None] for an effect. *)
let c_type cx (ty : Ssa_type.t) =
  match ty with
  | Ssa_type.Effect -> None
  | Ssa_type.Local -> Some ("double *", "0")
  | Ssa_type.Mask l -> Some (vector_type cx ~elem:`Mask l, "{0}")
  | Ssa_type.Scalar Ssa_type.F32 ->
      cx.f32 <- true;
      Some ("float", "0")
  | Ssa_type.Scalar Ssa_type.F64 -> Some ("double", "0")
  | Ssa_type.Scalar (Ssa_type.I64 | Ssa_type.Index) -> Some ("int64_t", "0")
  | Ssa_type.Scalar Ssa_type.Offset ->
      invalid_arg "Ssa_c: a native byte offset has no kernel form"
  | Ssa_type.Scalar Ssa_type.Pred -> Some ("int", "0")
  | Ssa_type.Vec (Ssa_type.F32, l) ->
      cx.f32 <- true;
      Some (vector_type cx ~elem:`F32 l, "{0}")
  | Ssa_type.Vec (Ssa_type.F64, l) -> Some (vector_type cx ~elem:`F64 l, "{0}")
  | Ssa_type.Vec (_, _) -> invalid_arg "Ssa_c: a vector of a non-float"

let is_erased (v : Ssa_value.t) = Ssa_type.equal v.Ssa_value.ty Ssa_type.Effect

(* A value's variable, declared at function scope the first time it is defined. *)
let define cx (v : Ssa_value.t) =
  match c_type cx v.Ssa_value.ty with
  | None -> ""
  | Some (ty, init) ->
      let name = Printf.sprintf "v%d" cx.next_name in
      cx.next_name <- cx.next_name + 1;
      Hashtbl.replace cx.names (v.Ssa_value.id :> int) name;
      cx.decls <- (name, ty, init) :: cx.decls;
      name

let name cx (v : Ssa_value.t) =
  match Hashtbl.find_opt cx.names (v.Ssa_value.id :> int) with
  | Some n -> n
  | None -> invalid_arg "Ssa_c: a value used before it is defined"

let temp cx =
  let n = Printf.sprintf "t%d" cx.next_temp in
  cx.next_temp <- cx.next_temp + 1;
  n

(* A failure site: where a record's static part (which local) is looked up. *)
let site cx (f : Loop_ir.Loop_failure.t) =
  match cx.site_table with
  | None ->
      let k = cx.site_count in
      cx.sites <- f :: cx.sites;
      cx.site_count <- k + 1;
      k
  | Some table ->
      (* the first entry of the caller's table that names the same failure; a
         check the table has no entry for is one its own lowering proved can
         never fire, so its index is one past the table, which a decoder
         reports as a defect rather than as a failure *)
      let rec find i =
        if i >= Array.length table then i
        else if Loop_ir.Loop_failure.same_site table.(i) f then i
        else find (i + 1)
      in
      find 0

let vector_helper cx name text =
  if not (List.mem_assoc name cx.vector_helpers) then
    cx.vector_helpers <- cx.vector_helpers @ [ (name, text) ]

let buffer_index cx id =
  let rec go i = function
    | [] -> invalid_arg "Ssa_c: an undeclared buffer"
    | (b : Ssa_buffer.t) :: rest ->
        if Ssa_id.Buffer.equal b.Ssa_buffer.id id then i else go (i + 1) rest
  in
  go 0 cx.buffers

let buffer_name cx id = "b" ^ string_of_int (buffer_index cx id)

let find_buffer cx id =
  match
    List.find_opt
      (fun (b : Ssa_buffer.t) -> Ssa_id.Buffer.equal b.Ssa_buffer.id id)
      cx.buffers
  with
  | Some b -> b
  | None -> invalid_arg "Ssa_c: an undeclared buffer"

(* The storage cell of a format, or a typed refusal. *)
let cell_type id (f : Ssa_format.t) =
  match f with
  | Ssa_format.Bf16 | Ssa_format.F16 -> "uint16_t"
  | Ssa_format.Bool -> "uint8_t"
  | Ssa_format.F32 -> "float"
  | Ssa_format.F64 -> "double"
  | Ssa_format.I32 -> "int32_t"
  | Ssa_format.I64 -> "int64_t"
  | Ssa_format.I16 _ | Ssa_format.I8 _ ->
      raise (Refused (`Unsupported_format (id, Ssa_format.name f)))

let i64_lit = Loop_ir.Loop_c_base.i64_lit
let float_lit = Loop_ir.Loop_c_base.float_lit
let int_lit n = Loop_ir.Loop_c_base.int_lit n
let fail_record = Loop_ir.Loop_c_base.fail_record
let outside_int32 = Loop_ir.Loop_c_base.outside_int32

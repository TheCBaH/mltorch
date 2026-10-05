open Ssa_ir
module I = Wasm.Instr
module R = Loop_ir.Loop_wasm_runtime
module W = Loop_ir.Loop_wasm_failure
module F = Loop_ir.Loop_js_failure
module LF = Loop_ir.Loop_failure

(* A kernel's instruction lists run to thousands of elements: the stock [@] and
   [List.map] recurse once per element, fine natively and a stack overflow under
   js_of_ocaml. *)
let ( @ ) a b = List.rev_append (List.rev a) b
let map f l = List.rev (List.rev_map f l)

type error =
  [ `Local_arrays_too_large of int64
  | `Loop_leaves_domain
  | `Unknown_site
  | `Unsupported_format of Ssa_id.Buffer.t * string
  | `Unsupported_lanes of int
  | `Unsupported_operation of string ]

let pp_error ppf : [< error ] -> unit = function
  | `Local_arrays_too_large n ->
      Fmt.pf ppf "local arrays need %Ld bytes, beyond the static region" n
  | `Unknown_site ->
      Fmt.string ppf "a failure site the program's site table does not name"
  | `Loop_leaves_domain ->
      Fmt.string ppf
        "a loop's last increment may leave the 32-bit index domain, so its \
         counter could wrap"
  | `Unsupported_format (b, f) ->
      Fmt.pf ppf "%a: format %s has no WebAssembly implementation"
        Ssa_id.Buffer.pp b f
  | `Unsupported_lanes n ->
      Fmt.pf ppf "a %d-lane vector does not fill 128-bit registers" n
  | `Unsupported_operation o -> Fmt.pf ppf "%s has no WebAssembly form" o

exception Refused of error

let refuse e = raise (Refused e)
let static_cap = 0x4000_0000L

type mask_shape =
  | Wide
  | Narrow
      (** The lanes a mask's registers hold: [Wide] is [i64x2] (the lanes of an
          [f64x2] comparison), [Narrow] is [i32x4] (an [f32x4] one). *)

type t = {
  mutable extra : Wasm_type.t list;  (** locals past the parameters, reversed *)
  n_params : int;
  values : (int, int array) Hashtbl.t;
      (** a value's locals, one per register *)
  mask_shapes : (int, mask_shape) Hashtbl.t;
  buffers : Ssa_buffer.t list;
  tables : (int, int * int) Hashtbl.t;  (** per-channel scale and zero tables *)
  locals : (int, int64 * Expr.Local_var.t option) Hashtbl.t;
      (** a scratch object's slots and the variable it names *)
  mutable local_top : int64;
  table_alloc : bytes:int -> int;
  mutable used : R.Callee.t list;
  mutable sites : LF.t list;
  mutable site_count : int;
  mutable meter : (int * int) option;  (** [scan_remaining], [scan_live] *)
  mutable f32 : bool;
  relaxed_madd : bool;
  ranges : Ssa_range.t;
  site_table : LF.t array option;
}

let n op = I.Numeric op
let i32 k = I.I32_const (Int32.of_int k)
let i64c k = I.I64_const k
let get i = I.Local_get i
let set i = I.Local_set i
let f64 x = I.F64_const (Int64.bits_of_float x)
let f32 x = I.F32_const (Int32.bits_of_float x)
let arg align offset = { Wasm.Mem_arg.align; offset }
let fits32 k = Int32.to_int (Int32.of_int k) = k
let fits32_64 (k : int64) = Int64.equal (Int64.of_int32 (Int64.to_int32 k)) k

let call st c =
  if not (List.mem c st.used) then st.used <- c :: st.used;
  R.call c

let fresh st ty =
  let id = st.n_params + List.length st.extra in
  st.extra <- ty :: st.extra;
  id

let site st f =
  match st.site_table with
  | None ->
      let k = st.site_count in
      st.sites <- f :: st.sites;
      st.site_count <- k + 1;
      k
  | Some table ->
      let rec find i =
        if i >= Array.length table then refuse `Unknown_site
        else if LF.same_site table.(i) f then i
        else find (i + 1)
      in
      find 0

(* An index literal that becomes an [i32] immediate. *)
let index_const k =
  if fits32_64 k then I.I32_const (Int64.to_int32 k)
  else invalid_arg "Ssa_wasm: an index constant outside 32 bits"

let is_erased (v : Ssa_value.t) = Ssa_type.equal v.Ssa_value.ty Ssa_type.Effect

(* ---- types ---------------------------------------------------------------------- *)

let lanes_per_register : Ssa_type.scalar -> int = function
  | Ssa_type.F32 -> 4
  | Ssa_type.F64 -> 2
  | Ssa_type.I64 | Ssa_type.Index | Ssa_type.Offset | Ssa_type.Pred ->
      invalid_arg "Ssa_wasm: a vector of a non-float"

(* How many registers a vector or mask type takes. A mask is counted in the
   registers of the shape it is held in, which the caller says. *)
let registers_of_lanes lanes per =
  let n = Ssa_type.Lanes.to_int lanes in
  if n < per || n mod per <> 0 then refuse (`Unsupported_lanes n);
  n / per

let local_types st (v : Ssa_value.t) ~shape : Wasm_type.t list =
  match v.Ssa_value.ty with
  | Ssa_type.Effect -> []
  | Ssa_type.Local -> [ Wasm_type.I32 ]
  | Ssa_type.Scalar Ssa_type.F32 ->
      st.f32 <- true;
      [ Wasm_type.F32 ]
  | Ssa_type.Scalar Ssa_type.F64 -> [ Wasm_type.F64 ]
  | Ssa_type.Scalar Ssa_type.I64 -> [ Wasm_type.I64 ]
  | Ssa_type.Scalar (Ssa_type.Index | Ssa_type.Pred) -> [ Wasm_type.I32 ]
  | Ssa_type.Scalar Ssa_type.Offset ->
      invalid_arg "Ssa_wasm: a native byte offset has no kernel form"
  | Ssa_type.Vec (s, l) ->
      if s = Ssa_type.F32 then st.f32 <- true;
      List.init
        (registers_of_lanes l (lanes_per_register s))
        (fun _ -> Wasm_type.V128)
  | Ssa_type.Mask l ->
      let per = match shape with Wide -> 2 | Narrow -> 4 in
      List.init (registers_of_lanes l per) (fun _ -> Wasm_type.V128)

(* Defines a value: one local per register. *)
let define ?(shape = Wide) st (v : Ssa_value.t) =
  match local_types st v ~shape with
  | [] -> [||]
  | tys ->
      let regs = Array.of_list (List.map (fresh st) tys) in
      Hashtbl.replace st.values (v.Ssa_value.id :> int) regs;
      (match v.Ssa_value.ty with
      | Ssa_type.Mask _ ->
          Hashtbl.replace st.mask_shapes (v.Ssa_value.id :> int) shape
      | _ -> ());
      regs

let regs st (v : Ssa_value.t) =
  match Hashtbl.find_opt st.values (v.Ssa_value.id :> int) with
  | Some r -> r
  | None -> invalid_arg "Ssa_wasm: a value used before it is defined"

let reg st v = (regs st v).(0)
let read st v = [ get (reg st v) ]

let mask_shape st (v : Ssa_value.t) =
  Option.value ~default:Wide
    (Hashtbl.find_opt st.mask_shapes (v.Ssa_value.id :> int))

(* ---- failures ------------------------------------------------------------------- *)

(* A failure is a returned status, never a trap: the record is written, then the
   kernel returns nonzero. Each field is an [i64] expression evaluated only here,
   once the check has fired. *)
let fail st kind (fields : (int * I.t list) list) =
  [ i32 (W.kind_index kind); call st R.Callee.Fail_set ]
  @ List.concat_map
      (fun (slot, value) ->
        [ i32 Loop_ir.Loop_wasm_ctx.error_address ]
        @ value
        @ [ I.Store (Wasm.Store.I64_store, arg 3 (W.slot_offset slot)) ])
      fields
  @ [ i32 1; I.Return ]

let i64_of_int k = [ I.I64_const (Int64.of_int k) ]
let wide l = l @ [ n Wasm_op.I64_extend_i32_s ]

(* [value] outside [-2^31, 2^31): shifted into [0, 2^32) it is above 2^32 - 1
   read as unsigned. *)
let outside_int32 value =
  value
  @ [
      I.I64_const 0x8000_0000L;
      n Wasm_op.I64_add;
      I.I64_const 0xFFFF_FFFFL;
      n Wasm_op.I64_gt_u;
    ]

let meter st =
  match st.meter with
  | Some m -> m
  | None ->
      let remaining = fresh st Wasm_type.I64 in
      let live = fresh st Wasm_type.I64 in
      st.meter <- Some (remaining, live);
      (remaining, live)

let meter_failure st which limit =
  fail st F.Kind.Scan_meter
    [
      ( 0,
        i64_of_int
          (match which with
          | F.Meter.State_over_limit -> 0
          | F.Meter.Updates_exhausted -> 1) );
      (1, [ I.I64_const limit ]);
    ]

(* ---- buffers -------------------------------------------------------------------- *)

let buffer_index st id =
  let rec go i = function
    | [] -> invalid_arg "Ssa_wasm: an undeclared buffer"
    | (b : Ssa_buffer.t) :: rest ->
        if Ssa_id.Buffer.equal b.Ssa_buffer.id id then i else go (i + 1) rest
  in
  go 0 st.buffers

(* Parameter 0 is the [local] base pointer, so buffer [k] is parameter [k + 1]. *)
let buffer_param st id = 1 + buffer_index st id

let find_buffer st id =
  match
    List.find_opt
      (fun (b : Ssa_buffer.t) -> Ssa_id.Buffer.equal b.Ssa_buffer.id id)
      st.buffers
  with
  | Some b -> b
  | None -> invalid_arg "Ssa_wasm: an undeclared buffer"

(* log2 of the cell width in bytes. *)
let cell_log2 (f : Ssa_format.t) =
  match f with
  | Ssa_format.Bool | Ssa_format.I8 _ -> 0
  | Ssa_format.Bf16 | Ssa_format.F16 | Ssa_format.I16 _ -> 1
  | Ssa_format.F32 | Ssa_format.I32 -> 2
  | Ssa_format.F64 | Ssa_format.I64 -> 3

let extents (b : Ssa_buffer.t) = Expr.Coord.to_list b.Ssa_buffer.extents

(* The element offset of a coordinate, row-major, in [i32]. *)
let coord_offset st (b : Ssa_buffer.t) (c : Ssa_value.t Expr.Coord.t) =
  match (Expr.Coord.to_list c, extents b) with
  | first :: rest, _ :: exts ->
      List.fold_left2
        (fun acc comp e ->
          acc
          @ [ index_const e; n Wasm_op.I32_mul ]
          @ read st comp
          @ [ n Wasm_op.I32_add ])
        (read st first) rest exts
  | _ -> invalid_arg "Ssa_wasm: a coordinate has six components"

let access_index st b = function
  | Ssa_access.Coord c -> coord_offset st b c
  | Ssa_access.Flat o -> read st o

(* The byte address of an element: the buffer's base plus the offset scaled by the
   cell width. The memory plan bounds every buffer, so this stays in a positive
   [int32]; the check that the cell is in range is a failure check. *)
let cell_address st (b : Ssa_buffer.t) index_instrs =
  let log2 = cell_log2 b.Ssa_buffer.format in
  [ get (buffer_param st b.Ssa_buffer.id) ]
  @ index_instrs
  @ (if log2 = 0 then [] else [ i32 log2; n Wasm_op.I32_shl ])
  @ [ n Wasm_op.I32_add ]

let cell_type id (f : Ssa_format.t) =
  match f with
  | Ssa_format.I16 _ | Ssa_format.I8 _ ->
      (* quantized cells are decoded, but only per tensor or per channel *)
      ignore id;
      ()
  | _ -> ()

let align8 x = Int64.logand (Int64.add x 7L) (Int64.lognot 7L)

(* Reserves [bytes] of the kernel's local region, 8-aligned, relative to the
   [local] base: invocations run one after another, so every kernel's arrays
   overlay the same region. *)
let reserve_local st bytes =
  let off = align8 st.local_top in
  let top = Int64.add off bytes in
  if Int64.compare top static_cap > 0 then refuse (`Local_arrays_too_large top);
  st.local_top <- top;
  Int64.to_int off

(* Instructions are a tree of structured blocks over flat instruction lists, as
   the binary format is. Nothing here is typed beyond its shape: [Wasm_check]
   proves operand types, labels and indices before [Wasm_encode] writes a
   byte. Constants carry their exact bits, so a NaN payload or [-0] cannot be
   lost to a host float conversion. *)

module Block_type = struct
  (* No block takes parameters in this subset: a block yields nothing or one
     value. *)
  type t = Wasm_type.t option
end

module Mem_arg = struct
  (* [align] is the log2 alignment hint; [offset] is a static byte offset in
     [0, 2^31) (the spec allows 2^32; the policy is tighter, so it fits a
     positive [int32] under js_of_ocaml). *)
  type t = { align : int; offset : int }
end

module Load = struct
  type t =
    | F32_load
    | F64_load
    | I32_load
    | I32_load16_s
    | I32_load16_u
    | I32_load8_s
    | I32_load8_u
    | I64_load

  let value_type = function
    | F32_load -> Wasm_type.F32
    | F64_load -> Wasm_type.F64
    | I32_load | I32_load16_s | I32_load16_u | I32_load8_s | I32_load8_u ->
        Wasm_type.I32
    | I64_load -> Wasm_type.I64

  (* The natural alignment, as a log2. *)
  let natural_align = function
    | F32_load | I32_load -> 2
    | F64_load | I64_load -> 3
    | I32_load16_s | I32_load16_u -> 1
    | I32_load8_s | I32_load8_u -> 0

  let byte = function
    | F32_load -> 0x2A
    | F64_load -> 0x2B
    | I32_load -> 0x28
    | I32_load16_s -> 0x2E
    | I32_load16_u -> 0x2F
    | I32_load8_s -> 0x2C
    | I32_load8_u -> 0x2D
    | I64_load -> 0x29

  let name = function
    | F32_load -> "f32.load"
    | F64_load -> "f64.load"
    | I32_load -> "i32.load"
    | I32_load16_s -> "i32.load16_s"
    | I32_load16_u -> "i32.load16_u"
    | I32_load8_s -> "i32.load8_s"
    | I32_load8_u -> "i32.load8_u"
    | I64_load -> "i64.load"
end

module Store = struct
  type t =
    | F32_store
    | F64_store
    | I32_store
    | I32_store16
    | I32_store8
    | I64_store

  let value_type = function
    | F32_store -> Wasm_type.F32
    | F64_store -> Wasm_type.F64
    | I32_store | I32_store16 | I32_store8 -> Wasm_type.I32
    | I64_store -> Wasm_type.I64

  let natural_align = function
    | F32_store | I32_store -> 2
    | F64_store | I64_store -> 3
    | I32_store16 -> 1
    | I32_store8 -> 0

  let byte = function
    | F32_store -> 0x38
    | F64_store -> 0x39
    | I32_store -> 0x36
    | I32_store16 -> 0x3B
    | I32_store8 -> 0x3A
    | I64_store -> 0x37

  let name = function
    | F32_store -> "f32.store"
    | F64_store -> "f64.store"
    | I32_store -> "i32.store"
    | I32_store16 -> "i32.store16"
    | I32_store8 -> "i32.store8"
    | I64_store -> "i64.store"
end

module Instr = struct
  type t =
    | Block of Block_type.t * t list
    | Br of int
    | Br_if of int
    | Call of int
    | Drop
    | F32_const of int32  (** the exact bits *)
    | F64_const of int64  (** the exact bits *)
    | Global_get of int
    | Global_set of int
    | I32_const of int32
    | I64_const of int64
    | If of Block_type.t * t list * t list
    | Load of Load.t * Mem_arg.t
    | Local_get of int
    | Local_set of int
    | Local_tee of int
    | Loop of Block_type.t * t list
    | Memory_copy  (** [dst src len], bytes, memory 0 *)
    | Memory_fill  (** [dst value len], bytes, memory 0 *)
    | Numeric of Wasm_op.t
    | Return
    | Select  (** the untyped form: numeric operands only *)
    | Store of Store.t * Mem_arg.t
    | Unreachable

  (* Renumbers every [Call], so a producer can emit calls to symbolic callees
     and fix the function index space once the module's imports and helpers
     are known. *)
  let rec map_calls f = function
    | Block (bt, l) -> Block (bt, List.map (map_calls f) l)
    | Call n -> Call (f n)
    | If (bt, yes, no) ->
        If (bt, List.map (map_calls f) yes, List.map (map_calls f) no)
    | Loop (bt, l) -> Loop (bt, List.map (map_calls f) l)
    | ( Br _ | Br_if _ | Drop | F32_const _ | F64_const _ | Global_get _
      | Global_set _ | I32_const _ | I64_const _ | Load _ | Local_get _
      | Local_set _ | Local_tee _ | Memory_copy | Memory_fill | Numeric _
      | Return | Select | Store _ | Unreachable ) as i ->
        i
end

module Func_type = struct
  type t = { params : Wasm_type.t list; results : Wasm_type.t list }

  let equal (a : t) (b : t) = a = b
end

module Import = struct
  (* A function import. *)
  type t = { module_name : string; name : string; type_ : Func_type.t }
end

module Func = struct
  type t = {
    type_ : Func_type.t;
    locals : Wasm_type.t list;  (** beyond the parameters *)
    body : Instr.t list;
  }
end

module Global = struct
  (* The initializer is a single constant instruction. *)
  type t = { type_ : Wasm_type.t; mutable_ : bool; init : Instr.t }
end

module Export = struct
  type kind = Func of int | Global of int | Memory
  type t = { name : string; kind : kind }
end

module Data = struct
  (* An active segment in memory 0 at a constant byte [offset] in [0, 2^31). *)
  type t = { offset : int; bytes : string }
end

module Memory = struct
  type t = { min_pages : int; max_pages : int option }

  let page_bytes = 65536
end

module Custom = struct
  type t = { name : string; payload : string }
end

module Module = struct
  (* Function indices count [imports] first, then [funcs]. *)
  type t = {
    imports : Import.t list;
    funcs : Func.t list;
    globals : Global.t list;
    memory : Memory.t option;
    exports : Export.t list;
    data : Data.t list;
    customs : Custom.t list;
  }

  let empty =
    {
      imports = [];
      funcs = [];
      globals = [];
      memory = None;
      exports = [];
      data = [];
      customs = [];
    }
end

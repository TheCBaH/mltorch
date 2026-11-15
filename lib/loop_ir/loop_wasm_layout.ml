module Width = struct
  type t = W1 | W2 | W4 | W8

  let log2 = function W1 -> 0 | W2 -> 1 | W4 -> 2 | W8 -> 3
  let arg w = { Wasm.Mem_arg.align = log2 w; offset = 0 }
  let bytes w = Int64.shift_left 1L (log2 w)
  let of_load l = { Wasm.Mem_arg.align = Wasm.Load.natural_align l; offset = 0 }

  let scale = function
    | W1 -> []
    | w ->
        [
          Wasm.Instr.I32_const (Int32.of_int (log2 w));
          Wasm.Instr.Numeric Wasm_op.I32_shl;
        ]
end

module type S = sig
  type t

  val zero : t
  val of_int64 : int64 -> t
  val pp : Format.formatter -> t -> unit
  val to_int64 : t -> int64
  val const : t -> Wasm.Instr.t
end

module Make () : sig
  include S

  val to_int : t -> int
end = struct
  type t = int

  let zero = 0

  let of_int64 k =
    if Int64.compare k 0L < 0 || Int64.compare k 0x8000_0000L >= 0 then
      invalid_arg "Loop_wasm_layout: a position outside [0, 2^31)"
    else Int64.to_int k

  let const k = Wasm.Instr.I32_const (Int32.of_int k)
  let to_int k = k
  let pp = Format.pp_print_int
  let to_int64 = Int64.of_int
end

module Address = struct
  include Make ()

  let data a bytes = { Wasm.Data.offset = to_int a; bytes }
  let advance a ~bytes = of_int64 (Int64.add (to_int64 a) bytes)
end

module Local_offset = Make ()

module Local = struct
  include
    Core.Tagged_int.Make
      (struct
        let prefix = "local"
      end)
      ()

  let base = of_int 0
end

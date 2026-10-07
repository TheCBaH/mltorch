(* The data model and checked layout arithmetic. Sizes, offsets and alignments
   are [int64] byte counts; an operation that would leave the non-negative
   signed 64-bit range is [None], never a wrapped value. A host array index is
   narrowed from these only after a check that it fits. *)

(* The one admitted data model: 64-bit pointers, little-endian, Linux/ELF. A
   generic program records it without naming an ISA; other pointer widths or
   byte orders are future, explicit extensions. *)
module Data_model = struct
  type t = Lp64_le

  let pointer_bytes Lp64_le = 8L
  let name Lp64_le = "lp64-le-elf"
  let equal (a : t) b = a = b
end

let add a b =
  if Int64.compare a 0L < 0 || Int64.compare b 0L < 0 then None
  else if Int64.compare a (Int64.sub Int64.max_int b) > 0 then None
  else Some (Int64.add a b)

let mul a b =
  if Int64.compare a 0L < 0 || Int64.compare b 0L < 0 then None
  else if Int64.equal a 0L || Int64.equal b 0L then Some 0L
  else if Int64.compare a (Int64.div Int64.max_int b) > 0 then None
  else Some (Int64.mul a b)

let is_power_of_two x =
  Int64.compare x 0L > 0 && Int64.equal (Int64.logand x (Int64.pred x)) 0L

(* [x] rounded up to a power-of-two [align]. *)
let align_up x ~align =
  if not (is_power_of_two align) then None
  else
    Option.map
      (fun y -> Int64.logand y (Int64.lognot (Int64.pred align)))
      (add x (Int64.pred align))

let aligned x ~align =
  is_power_of_two align && Int64.equal (Int64.logand x (Int64.pred align)) 0L

(* The largest alignment the interpreter and the admitted targets honor. *)
let max_align = 4096L

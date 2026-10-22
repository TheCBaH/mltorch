module Domain = struct
  type t = Imm12 | Li | Shamt | Simm5 | U20

  let pp ppf = function
    | Imm12 -> Fmt.string ppf "a signed 12-bit immediate"
    | Li -> Fmt.string ppf "a li immediate"
    | Shamt -> Fmt.string ppf "a shift amount"
    | Simm5 -> Fmt.string ppf "a signed 5-bit immediate"
    | U20 -> Fmt.string ppf "an unsigned 20-bit immediate"
end

module Imm_range = struct
  type t = { domain : Domain.t; value : int }

  let pp ppf { domain; value } =
    Fmt.pf ppf "%d is not %a" value Domain.pp domain
end

module type S = sig
  type t

  val domain : Domain.t
  val equal : t -> t -> bool
  val in_range : t -> bool
  val of_int : int -> t
  val pp : t Fmt.t
  val range : t -> Imm_range.t
  val to_int : t -> int
end

module Make (D : sig
  val domain : Domain.t
  val in_range : int -> bool
end) =
struct
  type t = int

  let domain = D.domain
  let equal = Int.equal
  let in_range = D.in_range
  let of_int n = n
  let pp = Fmt.int
  let range value = { Imm_range.domain = D.domain; value }
  let to_int n = n
end

let within lo hi n = n >= lo && n <= hi

module Imm12 = struct
  include Make (struct
    let domain = Domain.Imm12
    let in_range = within (-2048) 2047
  end)

  let minus_one = -1
  let zero = 0
end

module Li = Make (struct
  let domain = Domain.Li

  (* Probed against Rivet: [li] at 2047, 4096 and 0x7FFFF000 plans, 2048, 4097,
     -2049, 0x80000000 and 0xFFFFF000 are refused with an out-of-range [addi]. *)
  let in_range n =
    within (-2048) 2047 n
    || (n land 0xFFF = 0 && within (-0x8000_0000) 0x7FFF_F000 n)
end)

module Shamt = Make (struct
  let domain = Domain.Shamt
  let in_range = within 0 31
end)

module Simm5 = struct
  include Make (struct
    let domain = Domain.Simm5
    let in_range = within (-16) 15
  end)

  let zero = 0
end

module U20 = Make (struct
  let domain = Domain.U20
  let in_range = within 0 0xFFFFF
end)

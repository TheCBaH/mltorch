(* The vector configuration [vsetvli] sets. Sew and Lmul keep encoding order
   (SEW 8, 16, 32; LMUL 1, 2, 4, 8), which is also their numeric order. *)

module Sew = struct
  type t = E8 | E16 | E32

  let bits = function E8 -> 8 | E16 -> 16 | E32 -> 32

  let of_bits = function
    | 8 -> Some E8
    | 16 -> Some E16
    | 32 -> Some E32
    | _ -> None

  let pp ppf s = Fmt.pf ppf "e%d" (bits s)
end

module Lmul = struct
  type t = M1 | M2 | M4 | M8

  let factor = function M1 -> 1 | M2 -> 2 | M4 -> 4 | M8 -> 8

  let of_factor = function
    | 1 -> Some M1
    | 2 -> Some M2
    | 4 -> Some M4
    | 8 -> Some M8
    | _ -> None

  let pp ppf l = Fmt.pf ppf "m%d" (factor l)
end

module Policy = struct
  type t = Agnostic | Undisturbed

  (* [ta]/[tu] for the tail, [ma]/[mu] for the mask. *)
  let pp ~prefix ppf = function
    | Agnostic -> Fmt.pf ppf "%sa" prefix
    | Undisturbed -> Fmt.pf ppf "%su" prefix
end

(* The application vector length. [Reg r] is [vsetvli zero, r, ...]: the
   hardware grants [vl] without writing it to a register. [Vlmax] is
   [vsetvli vl, zero, ...], VLMAX written to [vl] (which must not be [zero]:
   [rd = rs1 = x0] keeps the current [vl], a different instruction). *)
module Avl = struct
  type t =
    | Reg of Rv32_kernel_reg.Gpr.t
    | Vlmax of { vl : Rv32_kernel_reg.Gpr.t }
end

type t = { sew : Sew.t; lmul : Lmul.t; tail : Policy.t; mask : Policy.t }

let pp ppf { sew; lmul; tail; mask } =
  Fmt.pf ppf "%a, %a, %a, %a" Sew.pp sew Lmul.pp lmul (Policy.pp ~prefix:"t")
    tail (Policy.pp ~prefix:"m") mask

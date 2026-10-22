module Class = struct
  type t = Fpr | Gpr | Vreg

  let pp ppf = function
    | Fpr -> Fmt.string ppf "floating-point"
    | Gpr -> Fmt.string ppf "integer"
    | Vreg -> Fmt.string ppf "vector"
end

module Register_index = struct
  type t = { class_ : Class.t; index : int }

  let pp ppf { class_; index } =
    Fmt.pf ppf "there is no %a register %d" Class.pp class_ index
end

module type Bank = sig
  type t

  val all : t list
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val index : t -> int
  val of_index : int -> (t, [> `Register_index of Register_index.t ]) Err.t
  val pp : t Fmt.t
  val to_rivet : t -> Riscv32_encode.Reg.t
end

(* A register is its number; the class types are kept apart by the signature. *)
module Make (C : sig
  val class_ : Class.t
  val names : string array
  val rivet : int -> Riscv32_encode.Reg.t
end) =
struct
  type t = int

  let all = List.init 32 Fun.id
  let compare = Int.compare
  let equal = Int.equal
  let index r = r

  let of_index index =
    if index >= 0 && index < 32 then Ok index
    else Err.fail (`Register_index { Register_index.class_ = C.class_; index })

  let pp ppf r = Fmt.string ppf C.names.(r)
  let to_rivet = C.rivet
end

module Gpr = struct
  include Make (struct
    let class_ = Class.Gpr

    let names =
      [|
        "zero";
        "ra";
        "sp";
        "gp";
        "tp";
        "t0";
        "t1";
        "t2";
        "s0";
        "s1";
        "a0";
        "a1";
        "a2";
        "a3";
        "a4";
        "a5";
        "a6";
        "a7";
        "s2";
        "s3";
        "s4";
        "s5";
        "s6";
        "s7";
        "s8";
        "s9";
        "s10";
        "s11";
        "t3";
        "t4";
        "t5";
        "t6";
      |]

    let rivet n = Riscv32_encode.Reg.X n
  end)

  let zero = 0
  let ra = 1
  let sp = 2
  let gp = 3
  let tp = 4
  let t0 = 5
  let t1 = 6
  let t2 = 7
  let s0 = 8
  let s1 = 9
  let a0 = 10
  let a1 = 11
  let a2 = 12
  let a3 = 13
  let a4 = 14
  let a5 = 15
  let a6 = 16
  let a7 = 17
  let s2 = 18
  let s3 = 19
  let s4 = 20
  let s5 = 21
  let s6 = 22
  let s7 = 23
  let s8 = 24
  let s9 = 25
  let s10 = 26
  let s11 = 27
  let t3 = 28
  let t4 = 29
  let t5 = 30
  let t6 = 31
end

module Fpr = struct
  include Make (struct
    let class_ = Class.Fpr

    let names =
      [|
        "ft0";
        "ft1";
        "ft2";
        "ft3";
        "ft4";
        "ft5";
        "ft6";
        "ft7";
        "fs0";
        "fs1";
        "fa0";
        "fa1";
        "fa2";
        "fa3";
        "fa4";
        "fa5";
        "fa6";
        "fa7";
        "fs2";
        "fs3";
        "fs4";
        "fs5";
        "fs6";
        "fs7";
        "fs8";
        "fs9";
        "fs10";
        "fs11";
        "ft8";
        "ft9";
        "ft10";
        "ft11";
      |]

    let rivet n = Riscv32_encode.Reg.F n
  end)

  let ft0 = 0
  let ft1 = 1
  let ft2 = 2
  let ft3 = 3
  let ft4 = 4
  let ft5 = 5
  let ft6 = 6
  let ft7 = 7
  let fs0 = 8
  let fs1 = 9
  let fa0 = 10
  let fa1 = 11
  let fa2 = 12
  let fa3 = 13
  let fa4 = 14
  let fa5 = 15
  let fa6 = 16
  let fa7 = 17
  let fs2 = 18
  let fs3 = 19
  let fs4 = 20
  let fs5 = 21
  let fs6 = 22
  let fs7 = 23
  let fs8 = 24
  let fs9 = 25
  let fs10 = 26
  let fs11 = 27
  let ft8 = 28
  let ft9 = 29
  let ft10 = 30
  let ft11 = 31
end

module Vreg = struct
  include Make (struct
    let class_ = Class.Vreg

    let names =
      [|
        "v0";
        "v1";
        "v2";
        "v3";
        "v4";
        "v5";
        "v6";
        "v7";
        "v8";
        "v9";
        "v10";
        "v11";
        "v12";
        "v13";
        "v14";
        "v15";
        "v16";
        "v17";
        "v18";
        "v19";
        "v20";
        "v21";
        "v22";
        "v23";
        "v24";
        "v25";
        "v26";
        "v27";
        "v28";
        "v29";
        "v30";
        "v31";
      |]

    let rivet n = Riscv32_encode.Reg.V n
  end)

  let v0 = 0
  let v1 = 1
  let v2 = 2
  let v3 = 3
  let v4 = 4
  let v5 = 5
  let v6 = 6
  let v7 = 7
  let v8 = 8
  let v9 = 9
  let v10 = 10
  let v11 = 11
  let v12 = 12
  let v13 = 13
  let v14 = 14
  let v15 = 15
  let v16 = 16
  let v17 = 17
  let v18 = 18
  let v19 = 19
  let v20 = 20
  let v21 = 21
  let v22 = 22
  let v23 = 23
  let v24 = 24
  let v25 = 25
  let v26 = 26
  let v27 = 27
  let v28 = 28
  let v29 = 29
  let v30 = 30
  let v31 = 31
end

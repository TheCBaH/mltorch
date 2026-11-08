(* The admitted Zve32f vector forms (the RISC-V V extension, 1.0) over virtual
   values, each under the vector configuration its lanes need, which the
   emitter establishes — SEW 32, tail and mask agnostic, LMUL 1 and vl 4 for
   four lanes, LMUL 4 and vl 16 for sixteen — so a vector is four binary32 or
   int32 lanes of a register, or sixteen of an aligned group of four, whatever
   VLEN beyond 128 the core has. A lane insert or extract, an extension and a
   narrow integer load exist at four lanes only. A mask is four lanes of all ones or all zeros, as a compare leaves
   them; RVV's own masks (one bit a lane, always in v0) exist only inside a
   form's sequence, which uses v0 in passing (and so clobbers it and admits no
   operand there, [Rv32_op.vector_scratch]). A strided access's
   stride is a register; a load or store takes its address whole. Every
   consumer matches this type exhaustively and alphabetically. *)

open Machine_ir
module Fcmp = Rv32_kind.Fcmp
module Fop = Rv32_kind.Fop
module Logic = Rv32_kind.Logic
module Sgnj = Rv32_kind.Sgnj

type v = Mir_value.t

type t =
  | Vadd of v * v  (** [vadd.vv]: int32 lanes, modulo 2^32 *)
  | Vcast of Mir_type.t * v
      (** [vmv.v.v]: the lanes' bits at another lane type (binary32 and int32)
      *)
  | Vext of { signed : bool; from : Mir_width.t; first : int; src : v }
      (** sequence: lanes [first .. first + 3] of an [i8x16] or [i16x8] register
          ([vslidedown.vi] at their width into v0 when [first] is not 0), sign-
          or zero-extended to four int32 lanes ([vsext.vf4] ... [vzext.vf2]);
          the result shares no register with the source *)
  | Vfbin of Fop.t * v * v  (** [vfadd.vv] ... [vfmax.vv] (maximumNumber) *)
  | Vfcmp of Fcmp.t * v * v
      (** sequence: [vmf<cmp>.vv] into v0, then all ones where it holds *)
  | Vfcvt_f_x of v  (** [vfcvt.f.x.v]: int32 lanes to binary32 *)
  | Vfcvt_x_f of v  (** [vfcvt.rtz.x.f.v]: toward zero, saturating *)
  | Vfmacc of v * v * v
      (** [vfmacc.vv]: [acc + a * b] a lane, one rounding, tied to the
          accumulator *)
  | Vfredosum of v * v
      (** sequence: [vfmv.s.f] of the seed into v0, [vfredosum.vs] of the vector
          into it, [vfmv.f.s]: the seed plus every lane in lane order, each
          addition rounded *)
  | Vfmv_f_s of int * v
      (** sequence: [vslidedown.vi] into v0 then [vfmv.f.s]: a lane *)
  | Vfmv_v_f of Mir_type.t * v
      (** [vfmv.v.f]: a binary32 to every lane of a vector of that type *)
  | Vfsqrt of v
  | Vinsert of int * v * v
      (** sequence: the vector with one lane replaced by the binary32, through
          [vfmv.s.f] into v0 and a tail-undisturbed [vslideup.vi] *)
  | Vle32 of Mir_type.t * v  (** [vle32.v]: four lanes from the address *)
  | Vle_int of Mir_width.t * int * v
      (** sequence: [lanes] i8 or i16 lanes ([vle8.v], [vle16.v]), a whole
          register or sixteen 16-bit lanes (half a group, at LMUL 2, only ever
          widened), under their own configuration, which is then restored *)
  | Vlogic of Logic.t * v * v  (** on masks or int32 lanes *)
  | Vlse32 of Mir_type.t * v * v
      (** [vlse32.v]: lane [k] at the address plus [k] times the stride *)
  | Vmerge of v * v * v
      (** sequence: [vmsne.vi] of the mask into v0, then [vmerge.vvm]: else,
          then, mask *)
  | Vmv of v  (** [vmv.v.v]: a whole vector or mask *)
  | Vmv_v_x of Mir_type.t * v
      (** [vmv.v.x]: a word to every lane, as int32 lanes or a mask *)
  | Vnot of v  (** [vxor.vi] with -1: a mask's complement *)
  | Vnsrl16 of v
      (** sequence: [vnsrl.wi] by 0 at SEW 16, LMUL 2: a group's sixteen int32
          lanes each truncated to sixteen bits, into half a group, under their
          own configuration, which is then restored *)
  | Vnsrl16_pair of v * v
      (** sequence: [vnsrl.wi] by 0 at SEW 16, LMUL 1/2, of the first register
          into the result and of the second into v0, then a tail-undisturbed
          [vslideup.vi] by 4 at LMUL 1: two registers' int32 lanes truncated to
          one register of eight 16-bit lanes; the result shares no register with
          an operand *)
  | Vse32 of v * v  (** [vse32.v]: address, value *)
  | Vse_int of Mir_width.t * int * v * v
      (** sequence: [lanes] i16 lanes ([vse16.v]), a whole register or half a
          group at LMUL 2, under their own configuration, which is then
          restored: address, value *)
  | Vshift of Rv32_kind.Shift.t * int * v
      (** [vsll.vi], [vsrl.vi], [vsra.vi] on int32 lanes *)
  | Vsgnj of Sgnj.t * v * v
  | Vsse32 of v * v * v  (** [vsse32.v]: address, stride, value *)

let uses = function
  | Vadd (a, b)
  | Vfbin (_, a, b)
  | Vfcmp (_, a, b)
  | Vlogic (_, a, b)
  | Vnsrl16_pair (a, b)
  | Vsgnj (_, a, b) ->
      [ a; b ]
  | Vcast (_, a)
  | Vext { src = a; _ }
  | Vle_int (_, _, a)
  | Vshift (_, _, a)
  | Vfcvt_f_x a
  | Vfcvt_x_f a
  | Vfmv_f_s (_, a)
  | Vfmv_v_f (_, a)
  | Vfsqrt a
  | Vle32 (_, a)
  | Vmv a
  | Vmv_v_x (_, a)
  | Vnot a
  | Vnsrl16 a ->
      [ a ]
  | Vfredosum (a, b)
  | Vinsert (_, a, b)
  | Vlse32 (_, a, b)
  | Vse32 (a, b)
  | Vse_int (_, _, a, b) ->
      [ a; b ]
  | Vfmacc (a, b, c) | Vmerge (a, b, c) | Vsse32 (a, b, c) -> [ a; b; c ]

let ordered = function
  | Vle32 _ | Vle_int _ | Vlse32 _ | Vse32 _ | Vse_int _ | Vsse32 _ -> true
  | _ -> false

let constraints = function
  | Vfmacc _ -> [ Mir_target.Constraint.Tied { result = 0; use = 0 } ]
  | Vext _ | Vnsrl16_pair _ -> [ Mir_target.Constraint.Early_clobber 0 ]
  | _ -> []

let lanes4 = Mir_type.Lanes.of_int 4
let f32x4 = Mir_type.Vec (Mir_type.Elem.F32, lanes4)
let i32x4 = Mir_type.Vec (Mir_type.Elem.Int Mir_width.W32, lanes4)
let mask4 = Mir_type.Mask lanes4

(* sixteen 16-bit lanes: half a group, only ever widened or narrowed into
   and stored *)
let i16x16 =
  Mir_type.Vec (Mir_type.Elem.Int Mir_width.W16, Mir_type.Lanes.of_int 16)

(* The lane counts a form may have: one register, or a group of four or of
   eight. *)
let lane_counts = [ 4; 16; 32 ]
let counted n = List.mem (Mir_type.Lanes.to_int n) lane_counts
let f32v n = Mir_type.Vec (Mir_type.Elem.F32, n)
let i32v n = Mir_type.Vec (Mir_type.Elem.Int Mir_width.W32, n)

(* The lanes of a binary32, int32 or mask vector a form admits. *)
let lanes_of = function
  | Mir_type.Vec ((Mir_type.Elem.F32 | Mir_type.Elem.Int Mir_width.W32), n)
  | Mir_type.Mask n
    when counted n ->
      Some n
  | _ -> None

let is_f32v t =
  match t with Mir_type.Vec (Mir_type.Elem.F32, n) -> counted n | _ -> false

let is_i32v t =
  match t with
  | Mir_type.Vec (Mir_type.Elem.Int Mir_width.W32, n) -> counted n
  | _ -> false

let is_mask t = match t with Mir_type.Mask n -> counted n | _ -> false

(* A whole register of i8 or i16 lanes. *)
let int_register = function
  | Mir_width.W8 ->
      Mir_type.Vec (Mir_type.Elem.Int Mir_width.W8, Mir_type.Lanes.of_int 16)
  | Mir_width.W16 ->
      Mir_type.Vec (Mir_type.Elem.Int Mir_width.W16, Mir_type.Lanes.of_int 8)
  | w -> invalid_arg ("Rv32_vop.int_register: " ^ Mir_width.name w)

let narrow_int w = w = Mir_width.W8 || w = Mir_width.W16
let ( let* ) = Result.bind
let ty (v : v) = v.Mir_value.ty
let is t u = Mir_type.equal t u
let need ok why = if ok then Ok () else Error why
let lane_ok k = k >= 0 && k < 4

let typing op =
  let fv a = is_f32v (ty a) in
  let same a b = is (ty a) (ty b) in
  match op with
  | Vadd (a, b) ->
      let* () = need (is_i32v (ty a) && same a b) "int32 lane operands" in
      Ok [ ty a ]
  | Vcast (t, a) ->
      let q x = is_f32v x || is_i32v x in
      let* () =
        need
          (q t && q (ty a) && lanes_of t = lanes_of (ty a))
          "a register's lanes"
      in
      Ok [ t ]
  | Vext { from; first; src; _ } ->
      let* () =
        need
          ((narrow_int from
           && is (ty src) (int_register from)
           && first >= 0
           && first mod 4 = 0
           && first + 4 <= if from = Mir_width.W8 then 16 else 8)
          || (from = Mir_width.W16 && is (ty src) i16x16 && first = 0))
          "extended lanes"
      in
      Ok
        [
          (if is (ty src) i16x16 then i32v (Mir_type.Lanes.of_int 16) else i32x4);
        ]
  | Vfbin (_, a, b) | Vsgnj (_, a, b) ->
      let* () = need (fv a && same a b) "vector operands" in
      Ok [ ty a ]
  | Vfcmp (_, a, b) ->
      let* () = need (fv a && same a b) "vector compare operands" in
      Ok [ Mir_type.Mask (Option.get (lanes_of (ty a))) ]
  | Vfcvt_f_x a ->
      let* () = need (is_i32v (ty a)) "int32 lanes" in
      Ok [ f32v (Option.get (lanes_of (ty a))) ]
  | Vfcvt_x_f a ->
      let* () = need (fv a) "binary32 lanes" in
      Ok [ i32v (Option.get (lanes_of (ty a))) ]
  | Vfmacc (acc, a, b) ->
      let* () = need (fv acc && same acc a && same acc b) "vfmacc operands" in
      Ok [ ty acc ]
  | Vfredosum (seed, a) ->
      let* () =
        need (is (ty seed) Mir_type.F32 && is_f32v (ty a)) "ordered reduction"
      in
      Ok [ Mir_type.F32 ]
  | Vfmv_f_s (k, a) ->
      (* a lane of one register, or of a group of four (the lane's own
         register, at LMUL 1) *)
      let* () =
        need
          (is_f32v (ty a)
          && k >= 0
          && k < Mir_type.Lanes.to_int (Option.get (lanes_of (ty a))))
          "lane extract"
      in
      Ok [ Mir_type.F32 ]
  | Vfmv_v_f (t, a) ->
      let* () = need (is (ty a) Mir_type.F32 && is_f32v t) "splat source" in
      Ok [ t ]
  | Vfsqrt a ->
      let* () = need (fv a) "vector operand" in
      Ok [ ty a ]
  | Vinsert (k, a, x) ->
      let* () =
        need
          (lane_ok k && is (ty a) f32x4 && is (ty x) Mir_type.F32)
          "lane insert"
      in
      Ok [ f32x4 ]
  | Vle32 (t, base) ->
      let* () = need (is (ty base) Mir_type.Ptr) "vle32 address" in
      let* () = need (is_f32v t || is_i32v t) "vle32 lanes" in
      Ok [ t ]
  | Vle_int (w, n, base) ->
      let* () = need (is (ty base) Mir_type.Ptr) "vle address" in
      let* () = need (narrow_int w) "integer register width" in
      let t = Mir_type.Vec (Mir_type.Elem.Int w, Mir_type.Lanes.of_int n) in
      let* () =
        need
          (is t (int_register w) || (w = Mir_width.W16 && n = 16))
          "integer register lanes"
      in
      Ok [ t ]
  | Vlogic (_, a, b) ->
      let* () =
        need
          ((is_mask (ty a) || is_i32v (ty a)) && same a b)
          "mask or int32 lane operands"
      in
      Ok [ ty a ]
  | Vlse32 (t, base, stride) ->
      let* () = need (is (ty base) Mir_type.Ptr) "vlse32 address" in
      let* () = need (is (ty stride) Mir_type.i32) "vlse32 stride" in
      let* () = need (is_f32v t || is_i32v t) "vlse32 lanes" in
      Ok [ t ]
  | Vmerge (other, a, m) ->
      let* () =
        need (is_mask (ty m) && lanes_of (ty m) = lanes_of (ty a)) "merge mask"
      in
      let* () =
        need
          ((fv a || is_i32v (ty a) || is_mask (ty a)) && same a other)
          "merge operands"
      in
      Ok [ ty a ]
  | Vmv a ->
      let* () =
        need (fv a || is_i32v (ty a) || is_mask (ty a)) "vector operand"
      in
      Ok [ ty a ]
  | Vmv_v_x (t, a) ->
      let* () = need (is (ty a) Mir_type.i32) "splat source" in
      let* () = need (is_i32v t || is_mask t) "splat lanes" in
      Ok [ t ]
  | Vnot a ->
      let* () = need (is_mask (ty a)) "mask operand" in
      Ok [ ty a ]
  | Vnsrl16 a ->
      let* () =
        need
          (is (ty a) (i32v (Mir_type.Lanes.of_int 16)))
          "a group's int32 lanes"
      in
      Ok [ i16x16 ]
  | Vnsrl16_pair (a, b) ->
      let* () = need (is (ty a) i32x4 && is (ty b) i32x4) "int32 lanes" in
      Ok [ int_register Mir_width.W16 ]
  | Vshift (o, k, a) ->
      let* () = need (is_i32v (ty a)) "shifted lanes" in
      let* () =
        need
          (k >= 0 && k < 32 && (k > 0 || o = Rv32_kind.Shift.Sll))
          "lane shift amount"
      in
      Ok [ ty a ]
  | Vse32 (base, x) ->
      let* () = need (is (ty base) Mir_type.Ptr) "vse32 address" in
      let* () = need (fv x || is_i32v (ty x)) "vse32 value" in
      Ok []
  | Vse_int (w, n, base, x) ->
      let* () = need (is (ty base) Mir_type.Ptr) "vse address" in
      let t = Mir_type.Vec (Mir_type.Elem.Int w, Mir_type.Lanes.of_int n) in
      let* () =
        need
          (w = Mir_width.W16 && (is t (int_register w) || n = 16) && is (ty x) t)
          "stored 16-bit lanes"
      in
      Ok []
  | Vsse32 (base, stride, x) ->
      let* () = need (is (ty base) Mir_type.Ptr) "vsse32 address" in
      let* () = need (is (ty stride) Mir_type.i32) "vsse32 stride" in
      let* () = need (fv x || is_i32v (ty x)) "vsse32 value" in
      Ok []

let pp pv fmt op =
  let vs = Fmt.(list ~sep:(any ", ") pv) in
  match op with
  | Vadd (a, b) -> Fmt.pf fmt "vadd.vv %a" vs [ a; b ]
  | Vcast (t, a) -> Fmt.pf fmt "vmv.v.v.%a %a" Mir_type.pp t pv a
  | Vext { signed; from; first; src } ->
      Fmt.pf fmt "v%cext.%s[%d] %a"
        (if signed then 's' else 'z')
        (Mir_width.name from) first pv src
  | Vfbin (o, a, b) -> Fmt.pf fmt "%s %a" (Fop.vname o) vs [ a; b ]
  | Vfredosum (s, a) -> Fmt.pf fmt "vfredosum.vs %a" vs [ s; a ]
  | Vfcmp (o, a, b) -> Fmt.pf fmt "%s.lanes %a" (Fcmp.vname o) vs [ a; b ]
  | Vfcvt_f_x a -> Fmt.pf fmt "vfcvt.f.x.v %a" pv a
  | Vfcvt_x_f a -> Fmt.pf fmt "vfcvt.rtz.x.f.v %a" pv a
  | Vfmacc (acc, a, b) -> Fmt.pf fmt "vfmacc.vv %a" vs [ acc; a; b ]
  | Vfmv_f_s (k, a) -> Fmt.pf fmt "vfmv.f.s %a[%d]" pv a k
  | Vfmv_v_f (t, a) -> Fmt.pf fmt "vfmv.v.f.%a %a" Mir_type.pp t pv a
  | Vfsqrt a -> Fmt.pf fmt "vfsqrt.v %a" pv a
  | Vinsert (k, a, x) -> Fmt.pf fmt "vinsert %a[%d], %a" pv a k pv x
  | Vle32 (t, base) -> Fmt.pf fmt "vle32.v.%a (%a)" Mir_type.pp t pv base
  | Vle_int (w, n, base) ->
      Fmt.pf fmt "vle%d.v x%d (%a)" (Mir_width.bits w) n pv base
  | Vlogic (o, a, b) -> Fmt.pf fmt "%s %a" (Logic.name o) vs [ a; b ]
  | Vlse32 (t, base, s) ->
      Fmt.pf fmt "vlse32.v.%a (%a), %a" Mir_type.pp t pv base pv s
  | Vmerge (other, a, m) -> Fmt.pf fmt "vmerge %a" vs [ other; a; m ]
  | Vmv a -> Fmt.pf fmt "vmv.v.v %a" pv a
  | Vmv_v_x (t, a) -> Fmt.pf fmt "vmv.v.x.%a %a" Mir_type.pp t pv a
  | Vnot a -> Fmt.pf fmt "vnot.v %a" pv a
  | Vnsrl16 a -> Fmt.pf fmt "vnsrl.wi.e16 %a" pv a
  | Vnsrl16_pair (a, b) -> Fmt.pf fmt "vnsrl.wi.e16.pair %a" vs [ a; b ]
  | Vse32 (base, x) -> Fmt.pf fmt "vse32.v %a, (%a)" pv x pv base
  | Vse_int (w, n, base, x) ->
      Fmt.pf fmt "vse%d.v x%d %a, (%a)" (Mir_width.bits w) n pv x pv base
  | Vshift (o, k, a) -> Fmt.pf fmt "%s %a, %d" (Rv32_kind.Shift.name o) pv a k
  | Vsgnj (o, a, b) -> Fmt.pf fmt "%s %a" (Sgnj.vname o) vs [ a; b ]
  | Vsse32 (base, s, x) -> Fmt.pf fmt "vsse32.v %a, (%a), %a" pv x pv base pv s

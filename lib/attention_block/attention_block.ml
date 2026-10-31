module Strides = struct
  type t = { batch : int64; head : int64; row : int64; col : int64 }
end

module Precision = struct
  type t = Bf16_paired | F32_fused | F32_ordered

  let all = [ Bf16_paired; F32_fused; F32_ordered ]

  let name = function
    | Bf16_paired -> "bf16_paired"
    | F32_fused -> "f32_fused"
    | F32_ordered -> "f32_ordered"

  let format = function
    | Bf16_paired -> Contraction.Element.Bf16
    | F32_fused | F32_ordered -> Contraction.Element.F32

  let numerics = function
    | Bf16_paired -> Contraction.Numerics.Paired
    | F32_fused -> Contraction.Numerics.Fused_sequential
    | F32_ordered -> Contraction.Numerics.Ordered
end

type t = {
  batch : int64;
  heads : int64;
  tq : int64;
  tk : int64;
  d : int64;
  q : Strides.t;
  k : Strides.t;
  v : Strides.t;
  o : Strides.t;
  mask : Strides.t option;
  scale : float;
  block_keys : int64;
  precision : Precision.t;
}

module Invalid = struct
  type t =
    | Block_keys of int64
    | Extent of { name : string; value : int64 }
    | Scale of float
    | Stride of { name : string; value : int64 }
end

let round32 = F32_math.round32

(* RISC-V arithmetic returns the canonical NaN whatever payload its operands
   carry; OCaml propagates it. Every binary32 result goes through here. *)
let nan32 = Int32.float_of_bits 0x7FC00000l
let r32 x = if Float.is_nan x then nan32 else round32 x

let make ~batch ~heads ~tq ~tk ~d ~q ~k ~v ~o ?mask ~scale ~block_keys
    ?(precision = Precision.Bf16_paired) () =
  let extent name value =
    if Int64.compare value 1L < 0 then
      Err.fail (`Invalid_attention (Invalid.Extent { name; value }))
    else Err.return ()
  and strides name (s : Strides.t) =
    let one field value =
      if Int64.compare value 0L < 0 then
        Err.fail
          (`Invalid_attention
             (Invalid.Stride { name = name ^ "." ^ field; value }))
      else Err.return ()
    in
    let open Err.Syntax in
    let* () = one "batch" s.Strides.batch in
    let* () = one "head" s.Strides.head in
    let* () = one "row" s.Strides.row in
    one "col" s.Strides.col
  in
  let open Err.Syntax in
  let* () = extent "batch" batch in
  let* () = extent "heads" heads in
  let* () = extent "tq" tq in
  let* () = extent "tk" tk in
  let* () = extent "d" d in
  let* () = strides "q" q in
  let* () = strides "k" k in
  let* () = strides "v" v in
  let* () = strides "o" o in
  let* () =
    match mask with Some m -> strides "mask" m | None -> Err.return ()
  in
  let* () =
    if
      Float.is_nan scale
      || Float.abs scale = Float.infinity
      || round32 scale <> scale
    then Err.fail (`Invalid_attention (Invalid.Scale scale))
    else Err.return ()
  in
  let* () =
    if Int64.compare block_keys 0L <= 0 || Int64.rem block_keys 4L <> 0L then
      Err.fail (`Invalid_attention (Invalid.Block_keys block_keys))
    else Err.return ()
  in
  Err.return
    { batch; heads; tq; tk; d; q; k; v; o; mask; scale; block_keys; precision }

let pp_strides ppf (s : Strides.t) =
  Fmt.pf ppf "(%Ld,%Ld,%Ld,%Ld)" s.Strides.batch s.Strides.head s.Strides.row
    s.Strides.col

let pp ppf t =
  Fmt.pf ppf
    "@[<h>attend batch=%Ld heads=%Ld tq=%Ld tk=%Ld d=%Ld q=%a k=%a v=%a o=%a \
     mask=%a scale=%h block_keys=%Ld%a@]"
    t.batch t.heads t.tq t.tk t.d pp_strides t.q pp_strides t.k pp_strides t.v
    pp_strides t.o
    Fmt.(option ~none:(any "none") pp_strides)
    t.mask t.scale t.block_keys
    (fun ppf -> function
      | Precision.Bf16_paired -> ()
      | p -> Fmt.pf ppf " precision=%s" (Precision.name p))
    t.precision

let pp_error ppf (`Invalid_attention i) =
  match i with
  | Invalid.Block_keys n ->
      Fmt.pf ppf "attention block of %Ld keys, not a positive multiple of four"
        n
  | Invalid.Extent { name; value } ->
      Fmt.pf ppf "attention extent %s = %Ld" name value
  | Invalid.Scale s -> Fmt.pf ppf "attention scale %h, not a finite binary32" s
  | Invalid.Stride { name; value } ->
      Fmt.pf ppf "attention stride %s = %Ld" name value

module Scale = struct
  let default ~d = round32 (1. /. sqrt (Int64.to_float d))
  let explicit = round32
end

let strides_text (s : Strides.t) =
  Printf.sprintf "%Ld_%Ld_%Ld_%Ld" s.Strides.batch s.Strides.head s.Strides.row
    s.Strides.col

(* The default precision adds no field, so a bfloat16 descriptor keeps the name,
   and with it the symbol, it had before the precision existed. *)
let name t =
  Printf.sprintf
    "attend.b%Ld.h%Ld.q%Ld.k%Ld.d%Ld.sq%s.sk%s.sv%s.so%s.%s.x%lx.n%Ld%s" t.batch
    t.heads t.tq t.tk t.d (strides_text t.q) (strides_text t.k)
    (strides_text t.v) (strides_text t.o)
    (match t.mask with None -> "nomask" | Some m -> "m" ^ strides_text m)
    (Int32.bits_of_float t.scale)
    t.block_keys
    (match t.precision with
    | Precision.Bf16_paired -> ""
    | p -> ".p" ^ Precision.name p)

let of_name s =
  let tagged tag field =
    let n = String.length tag in
    if String.length field > n && String.sub field 0 n = tag then
      Some (String.sub field n (String.length field - n))
    else None
  in
  let int_of tag field = Option.bind (tagged tag field) Int64.of_string_opt in
  let strides_of tag field =
    match tagged tag field with
    | None -> None
    | Some text -> (
        match List.map Int64.of_string_opt (String.split_on_char '_' text) with
        | [ Some batch; Some head; Some row; Some col ] ->
            Some { Strides.batch; head; row; col }
        | _ -> None)
  in
  let precision_of field =
    Option.bind (tagged "p" field) (fun text ->
        List.find_opt (fun p -> Precision.name p = text) Precision.all)
  in
  let fields =
    match String.split_on_char '.' s with
    | [ "attend"; b; h; q; k; d; sq; sk; sv; so; mask; x; n ] ->
        Some
          ( [ b; h; q; k; d; sq; sk; sv; so; mask; x; n ],
            Some Precision.Bf16_paired )
    | [ "attend"; b; h; q; k; d; sq; sk; sv; so; mask; x; n; p ] ->
        Some ([ b; h; q; k; d; sq; sk; sv; so; mask; x; n ], precision_of p)
    | _ -> None
  in
  match fields with
  | Some ([ b; h; q; k; d; sq; sk; sv; so; mask; x; n ], Some precision) -> (
      let mask =
        if mask = "nomask" then Some None
        else Option.map Option.some (strides_of "m" mask)
      in
      match
        ( int_of "b" b,
          int_of "h" h,
          int_of "q" q,
          int_of "k" k,
          int_of "d" d,
          strides_of "sq" sq,
          strides_of "sk" sk,
          strides_of "sv" sv,
          strides_of "so" so,
          mask,
          Option.bind (tagged "x" x) (fun x -> Int32.of_string_opt ("0x" ^ x)),
          int_of "n" n )
      with
      | ( Some batch,
          Some heads,
          Some tq,
          Some tk,
          Some d,
          Some q,
          Some k,
          Some v,
          Some o,
          Some mask,
          Some scale,
          Some block_keys ) ->
          Result.to_option
            (Err.payload
               (make ~batch ~heads ~tq ~tk ~d ~q ~k ~v ~o ?mask
                  ~scale:(Int32.float_of_bits scale)
                  ~block_keys ~precision ()))
      | _ -> None)
  | _ -> None

let symbol t =
  "mlt_attend_" ^ String.sub (Digest.to_hex (Digest.string (name t))) 0 16

(* bfloat16 round to nearest even of a binary32 value, on the bit pattern in
   two halves (js_of_ocaml's [int] has 32 bits): [x + 0x7FFF + bit 16], then
   the high half. A NaN stays NaN only when its low half is clear, as every
   arithmetic result's is. *)
let bf16_round x =
  let b = Int32.bits_of_float x in
  let hi = Int32.to_int (Int32.shift_right_logical b 16)
  and lo = Int32.to_int (Int32.logand b 0xFFFFl) in
  let carry = (lo + 0x7FFF + (hi land 1)) lsr 16 in
  Int32.float_of_bits
    (Int32.shift_left (Int32.of_int ((hi + carry) land 0xFFFF)) 16)

(* RISC-V [fmax]: a NaN operand is dropped unless both are NaN; +0 is above -0. *)
let fmax a b =
  if Float.is_nan a then if Float.is_nan b then nan32 else b
  else if Float.is_nan b then a
  else if a > b then a
  else if b > a then b
  else if Float.sign_bit a then b
  else a

let exp x = if Float.is_nan x then nan32 else F32_math.Value.exp x

(* an [int] index: [a * n + b] *)
let ix a n b = (a * n) + b

let eval t ~q ~k ~v ~mask ~write =
  let int = Int64.to_int in
  let each n f =
    let rec go i =
      if Int64.compare i n < 0 then (
        f i;
        go (Int64.succ i))
    in
    go 0L
  in
  (* sizes are bounded by the caller's arrays; the descriptor's extents are
     [int64] and narrow here, after the allocation would have failed first *)
  let tq = int t.tq and d = int t.d and bk = int t.block_keys in
  let ( + ) = Int64.add and ( * ) = Int64.mul in
  let bf16 = t.precision = Precision.Bf16_paired in
  let paired ~batch ~m ~n ~kk ~a ~b ~c ~seed =
    let desc =
      Result.get_ok
        (Err.payload
           (Contraction.make ~batch ~m ~n ~k:kk ~a ~b ~c
              ~format:(Precision.format t.precision)
              ~seed
              ~numerics:(Precision.numerics t.precision)))
    in
    Contraction.eval desc
  in
  each t.batch (fun bb ->
      each t.heads (fun h ->
          let q_base = (bb * t.q.Strides.batch) + (h * t.q.Strides.head)
          and k_base = (bb * t.k.Strides.batch) + (h * t.k.Strides.head)
          and v_base = (bb * t.v.Strides.batch) + (h * t.v.Strides.head)
          and o_base = (bb * t.o.Strides.batch) + (h * t.o.Strides.head) in
          (* per query column: the running maximum, sum and accumulator *)
          let m = Array.make tq Float.neg_infinity
          and l = Array.make tq 0.
          and acc = Array.make (ix tq d 0) 0. in
          let blocks =
            Int64.div (Int64.pred (t.tk + t.block_keys)) t.block_keys
          in
          each blocks (fun j ->
              let k0 = j * t.block_keys in
              let vk = int (Int64.min t.block_keys (Int64.sub t.tk k0)) in
              (* scores [vk][tq] *)
              let s = Array.make (ix vk tq 0) 0. in
              (paired ~batch:1L ~m:(Int64.of_int vk) ~n:t.tq ~kk:t.d
                 ~a:
                   {
                     Contraction.Operand.batch = 0L;
                     outer = t.k.Strides.row;
                     k = t.k.Strides.col;
                   }
                 ~b:
                   {
                     Contraction.Operand.batch = 0L;
                     outer = t.q.Strides.row;
                     k = t.q.Strides.col;
                   }
                 ~c:{ Contraction.Output.batch = 0L; row = t.tq; col = 1L }
                 ~seed:Contraction.Seed.Zero)
                ~a:(fun o -> k (k_base + (k0 * t.k.Strides.row) + o))
                ~b:(fun o -> q (q_base + o))
                ~c:(fun _ -> 0.)
                ~write:(fun o x -> s.(int o) <- x);
              (* z, then the new maximum, the rescale factor and p *)
              let pb = Array.make (ix tq bk 0) 0. in
              let seed = Array.make (ix tq d 0) 0. in
              for i = 0 to tq - 1 do
                let z =
                  Array.init vk (fun r ->
                      let sc = r32 (s.(ix r tq i) *. t.scale) in
                      match t.mask with
                      | None -> sc
                      | Some ms ->
                          r32
                            (sc
                            +. mask
                                 ((bb * ms.Strides.batch) + (h * ms.Strides.head)
                                 + (Int64.of_int i * ms.Strides.row)
                                 + ((k0 + Int64.of_int r) * ms.Strides.col))))
                in
                let m' =
                  Array.fold_left (if bf16 then fmax else Float.max) m.(i) z
                in
                let a =
                  if m' = Float.neg_infinity then 1.
                  else exp (r32 (m.(i) -. m'))
                in
                let sum = ref 0. in
                Array.iteri
                  (fun r zr ->
                    let p =
                      if zr = Float.neg_infinity then 0.
                      else exp (r32 (zr -. m'))
                    in
                    sum := r32 (!sum +. p);
                    pb.(ix i bk r) <- (if bf16 then bf16_round p else p))
                  z;
                m.(i) <- m';
                l.(i) <- r32 (r32 (l.(i) *. a) +. !sum);
                for f = 0 to d - 1 do
                  seed.(ix i d f) <- r32 (a *. acc.(ix i d f))
                done
              done;
              (* context: bfloat16 runs all [block_keys] keys, +0 past the valid
                 ones; binary32 runs the valid keys only *)
              let vb = Array.make (ix bk d 0) 0. in
              for r = 0 to vk - 1 do
                for f = 0 to d - 1 do
                  vb.(ix r d f) <-
                    v
                      (v_base
                      + ((k0 + Int64.of_int r) * t.v.Strides.row)
                      + (Int64.of_int f * t.v.Strides.col))
                done
              done;
              (paired ~batch:1L ~m:t.tq ~n:t.d
                 ~kk:(if bf16 then t.block_keys else Int64.of_int vk)
                 ~a:
                   {
                     Contraction.Operand.batch = 0L;
                     outer = t.block_keys;
                     k = 1L;
                   }
                 ~b:{ Contraction.Operand.batch = 0L; outer = 1L; k = t.d }
                 ~c:{ Contraction.Output.batch = 0L; row = t.d; col = 1L }
                 ~seed:Contraction.Seed.Tensor)
                ~a:(fun o -> pb.(int o))
                ~b:(fun o -> vb.(int o))
                ~c:(fun o -> seed.(int o))
                ~write:(fun o x -> acc.(int o) <- x));
          for i = 0 to tq - 1 do
            for f = 0 to d - 1 do
              write
                (o_base
                + (Int64.of_int i * t.o.Strides.row)
                + (Int64.of_int f * t.o.Strides.col))
                (if l.(i) = 0. then 0. else r32 (acc.(ix i d f) /. l.(i)))
            done
          done))

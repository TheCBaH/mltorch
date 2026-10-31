module P = Attention_block.Precision

module Kind = struct
  type t = Layer | Rms

  let name = function Layer -> "layer" | Rms -> "rms"
end

module Strides = struct
  type t = { row : int64; col : int64 }
end

type t = {
  kind : Kind.t;
  rows : int64;
  width : int64;
  eps : float;
  weight : bool;
  bias : bool;
  x : Strides.t;
  y : Strides.t;
  precision : P.t;
}

module Invalid = struct
  type t =
    | Bias_without_layer
    | Eps of float
    | Extent of { name : string; value : int64 }
    | Stride of { name : string; value : int64 }
end

let round32 = F32_math.round32
let nan32 = Int32.float_of_bits 0x7FC00000l
let r32 x = if Float.is_nan x then nan32 else round32 x

(* 2^24: the widest row whose count binary32 holds exactly *)
let max_width = 0x100_0000L

let make ~kind ~rows ~width ~eps ~weight ~bias ~x ~y ~precision =
  let fail i = Err.fail (`Invalid_norm i) in
  let extent name value =
    if Int64.compare value 1L < 0 then fail (Invalid.Extent { name; value })
    else Err.return ()
  and strides name (s : Strides.t) =
    let one field value =
      if Int64.compare value 0L < 0 then
        fail (Invalid.Stride { name = name ^ "." ^ field; value })
      else Err.return ()
    in
    let open Err.Syntax in
    let* () = one "row" s.Strides.row in
    one "col" s.Strides.col
  in
  let open Err.Syntax in
  let* () = extent "rows" rows in
  let* () = extent "width" width in
  let* () =
    if Int64.compare width max_width > 0 then
      fail (Invalid.Extent { name = "width"; value = width })
    else Err.return ()
  in
  let* () =
    if
      Float.is_nan eps
      || Float.abs eps = Float.infinity
      || round32 eps <> eps
      || (Float.sign_bit eps && eps <> 0.)
    then fail (Invalid.Eps eps)
    else Err.return ()
  in
  let* () =
    match (kind, bias) with
    | Kind.Rms, true -> fail Invalid.Bias_without_layer
    | Kind.Rms, false | Kind.Layer, (false | true) -> Err.return ()
  in
  let* () = strides "x" x in
  let* () = strides "y" y in
  Err.return { kind; rows; width; eps; weight; bias; x; y; precision }

let pp ppf t =
  Fmt.pf ppf
    "@[<h>norm %s rows=%Ld width=%Ld eps=%h weight=%b bias=%b x=(%Ld,%Ld) \
     y=(%Ld,%Ld) precision=%s@]"
    (Kind.name t.kind) t.rows t.width t.eps t.weight t.bias t.x.Strides.row
    t.x.Strides.col t.y.Strides.row t.y.Strides.col (P.name t.precision)

let pp_error ppf (`Invalid_norm i) =
  match i with
  | Invalid.Bias_without_layer -> Fmt.pf ppf "an RmsNorm has no bias"
  | Invalid.Eps e ->
      Fmt.pf ppf "norm eps %h, not a finite non-negative binary32" e
  | Invalid.Extent { name; value } ->
      Fmt.pf ppf "norm extent %s = %Ld" name value
  | Invalid.Stride { name; value } ->
      Fmt.pf ppf "norm stride %s = %Ld" name value

let affine t = (if t.weight then "w" else "") ^ if t.bias then "b" else ""

let name t =
  Printf.sprintf "norm.%s.r%Ld.w%Ld.e%lx.a%s.x%Ld_%Ld.y%Ld_%Ld.p%s"
    (Kind.name t.kind) t.rows t.width
    (Int32.bits_of_float t.eps)
    (affine t) t.x.Strides.row t.x.Strides.col t.y.Strides.row t.y.Strides.col
    (P.name t.precision)

let of_name s =
  let tagged tag field =
    let n = String.length tag in
    if String.length field >= n && String.sub field 0 n = tag then
      Some (String.sub field n (String.length field - n))
    else None
  in
  let int_of tag field = Option.bind (tagged tag field) Int64.of_string_opt in
  let strides_of tag field =
    match tagged tag field with
    | None -> None
    | Some text -> (
        match List.map Int64.of_string_opt (String.split_on_char '_' text) with
        | [ Some row; Some col ] -> Some { Strides.row; col }
        | _ -> None)
  in
  match String.split_on_char '.' s with
  | [ "norm"; kind; r; w; e; a; x; y; p ] -> (
      let kind =
        List.find_opt (fun k -> Kind.name k = kind) [ Kind.Layer; Kind.Rms ]
      and affine =
        match tagged "a" a with
        | Some "" -> Some (false, false)
        | Some "w" -> Some (true, false)
        | Some "b" -> Some (false, true)
        | Some "wb" -> Some (true, true)
        | Some _ | None -> None
      and precision =
        Option.bind (tagged "p" p) (fun text ->
            List.find_opt (fun p -> P.name p = text) P.all)
      in
      match
        ( kind,
          int_of "r" r,
          int_of "w" w,
          Option.bind (tagged "e" e) (fun e -> Int32.of_string_opt ("0x" ^ e)),
          affine,
          strides_of "x" x,
          strides_of "y" y,
          precision )
      with
      | ( Some kind,
          Some rows,
          Some width,
          Some eps,
          Some (weight, bias),
          Some x,
          Some y,
          Some precision ) ->
          Result.to_option
            (Err.payload
               (make ~kind ~rows ~width ~eps:(Int32.float_of_bits eps) ~weight
                  ~bias ~x ~y ~precision))
      | _ -> None)
  | _ -> None

let symbol t =
  "mlt_norm_" ^ String.sub (Digest.to_hex (Digest.string (name t))) 0 16

let fma32 a b c =
  if Float.is_nan a || Float.is_nan b || Float.is_nan c then nan32
  else
    let r = F32_math.fma32 a b c in
    if Float.is_nan r then nan32 else r

let eval t ~x ~weight ~bias ~write =
  let ( + ) = Int64.add and ( * ) = Int64.mul in
  let each n f =
    let rec go i =
      if Int64.compare i n < 0 then (
        f i;
        go (Int64.succ i))
    in
    go 0L
  in
  let fold n init f =
    let rec go i acc =
      if Int64.compare i n < 0 then go (Int64.succ i) (f acc i) else acc
    in
    go 0L init
  in
  let n = Int64.to_float t.width in
  let square acc v =
    match t.precision with
    | P.F32_fused -> fma32 v v acc
    | P.Bf16_paired | P.F32_ordered -> r32 (acc +. r32 (v *. v))
  in
  let out v =
    match t.precision with
    | P.Bf16_paired -> Attention_block.bf16_round v
    | P.F32_fused | P.F32_ordered -> v
  in
  each t.rows (fun r ->
      let xr = r * t.x.Strides.row and yr = r * t.y.Strides.row in
      let at c = x (xr + (c * t.x.Strides.col)) in
      let centre, sum =
        match t.kind with
        | Kind.Layer ->
            let s = fold t.width 0. (fun s c -> r32 (s +. at c)) in
            let mean = r32 (s /. n) in
            let centre c = r32 (at c -. mean) in
            (centre, fold t.width 0. (fun acc c -> square acc (centre c)))
        | Kind.Rms -> (at, fold t.width 0. (fun acc c -> square acc (at c)))
      in
      let inv = r32 (1. /. r32 (sqrt (r32 (r32 (sum /. n) +. t.eps)))) in
      each t.width (fun c ->
          let v = r32 (centre c *. inv) in
          let v = if t.weight then r32 (v *. weight c) else v in
          let v = if t.bias then r32 (v +. bias c) else v in
          write (yr + (c * t.y.Strides.col)) (out v)))

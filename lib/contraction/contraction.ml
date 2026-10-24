module Element = struct
  type t = Bf16 | F32

  let name = function Bf16 -> "bf16" | F32 -> "f32"
  let bytes = function Bf16 -> 2L | F32 -> 4L
end

module Seed = struct
  type t = Tensor | Zero

  let name = function Tensor -> "tensor" | Zero -> "zero"
end

module Numerics = struct
  type t = Fused_sequential | Ordered

  let name = function
    | Fused_sequential -> "fused_sequential"
    | Ordered -> "ordered"
end

module Operand = struct
  type t = { batch : int64; outer : int64; k : int64 }
end

module Output = struct
  type t = { batch : int64; row : int64; col : int64 }
end

type t = {
  batch : int64;
  m : int64;
  n : int64;
  k : int64;
  a : Operand.t;
  b : Operand.t;
  c : Output.t;
  format : Element.t;
  seed : Seed.t;
  numerics : Numerics.t;
}

module Invalid = struct
  type t =
    | Extent of { name : string; value : int64 }
    | Stride of { name : string; value : int64 }
end

let make ~batch ~m ~n ~k ~a ~b ~c ~format ~seed ~numerics =
  let extent name value least =
    if Int64.compare value least < 0 then
      Err.fail (`Invalid_contraction (Invalid.Extent { name; value }))
    else Err.return ()
  and stride name value =
    if Int64.compare value 0L < 0 then
      Err.fail (`Invalid_contraction (Invalid.Stride { name; value }))
    else Err.return ()
  in
  let open Err.Syntax in
  let* () = extent "batch" batch 1L in
  let* () = extent "m" m 1L in
  let* () = extent "n" n 1L in
  let* () = extent "k" k 0L in
  let* () = stride "a.batch" a.Operand.batch in
  let* () = stride "a.outer" a.Operand.outer in
  let* () = stride "a.k" a.Operand.k in
  let* () = stride "b.batch" b.Operand.batch in
  let* () = stride "b.outer" b.Operand.outer in
  let* () = stride "b.k" b.Operand.k in
  let* () = stride "c.batch" c.Output.batch in
  let* () = stride "c.row" c.Output.row in
  let* () = stride "c.col" c.Output.col in
  Err.return { batch; m; n; k; a; b; c; format; seed; numerics }

let pp ppf t =
  Fmt.pf ppf
    "@[<h>contract %s batch=%Ld m=%Ld n=%Ld k=%Ld a=(%Ld,%Ld,%Ld) \
     b=(%Ld,%Ld,%Ld) c=(%Ld,%Ld,%Ld) seed=%s %s@]"
    (Element.name t.format) t.batch t.m t.n t.k t.a.Operand.batch
    t.a.Operand.outer t.a.Operand.k t.b.Operand.batch t.b.Operand.outer
    t.b.Operand.k t.c.Output.batch t.c.Output.row t.c.Output.col
    (Seed.name t.seed) (Numerics.name t.numerics)

let pp_error ppf (`Invalid_contraction i) =
  match i with
  | Invalid.Extent { name; value } ->
      Fmt.pf ppf "contraction extent %s = %Ld" name value
  | Invalid.Stride { name; value } ->
      Fmt.pf ppf "contraction stride %s = %Ld" name value

let name t =
  Printf.sprintf
    "contract.%s.b%Ld.m%Ld.n%Ld.k%Ld.a%Ld_%Ld_%Ld.b%Ld_%Ld_%Ld.c%Ld_%Ld_%Ld.%s.%s"
    (Element.name t.format) t.batch t.m t.n t.k t.a.Operand.batch
    t.a.Operand.outer t.a.Operand.k t.b.Operand.batch t.b.Operand.outer
    t.b.Operand.k t.c.Output.batch t.c.Output.row t.c.Output.col
    (Seed.name t.seed) (Numerics.name t.numerics)

let of_name s =
  let int_of tag field =
    if String.length field > 0 && field.[0] = tag then
      Int64.of_string_opt (String.sub field 1 (String.length field - 1))
    else None
  in
  let triple tag field =
    if String.length field = 0 || field.[0] <> tag then None
    else
      match
        String.split_on_char '_' (String.sub field 1 (String.length field - 1))
      with
      | [ x; y; z ] -> (
          match
            (Int64.of_string_opt x, Int64.of_string_opt y, Int64.of_string_opt z)
          with
          | Some x, Some y, Some z -> Some (x, y, z)
          | _ -> None)
      | _ -> None
  in
  match String.split_on_char '.' s with
  | [ "contract"; format; b; m; n; k; a; bb; c; seed; numerics ] -> (
      match
        ( List.find_opt
            (fun x -> Element.name x = format)
            [ Element.Bf16; Element.F32 ],
          int_of 'b' b,
          int_of 'm' m,
          int_of 'n' n,
          int_of 'k' k,
          triple 'a' a,
          triple 'b' bb,
          triple 'c' c,
          List.find_opt (fun x -> Seed.name x = seed) [ Seed.Tensor; Seed.Zero ],
          List.find_opt
            (fun x -> Numerics.name x = numerics)
            [ Numerics.Fused_sequential; Numerics.Ordered ] )
      with
      | ( Some format,
          Some batch,
          Some m,
          Some n,
          Some k,
          Some (a0, a1, a2),
          Some (b0, b1, b2),
          Some (c0, c1, c2),
          Some seed,
          Some numerics ) ->
          Result.to_option
            (Err.payload
               (make ~batch ~m ~n ~k
                  ~a:{ Operand.batch = a0; outer = a1; k = a2 }
                  ~b:{ Operand.batch = b0; outer = b1; k = b2 }
                  ~c:{ Output.batch = c0; row = c1; col = c2 }
                  ~format ~seed ~numerics))
      | _ -> None)
  | _ -> None

let symbol t =
  "mlt_contract_" ^ String.sub (Digest.to_hex (Digest.string (name t))) 0 16

let macs t = Int64.mul (Int64.mul t.batch t.m) (Int64.mul t.n t.k)
let round32 = F32_math.round32

let eval t ~a ~b ~c ~write =
  let ( + ) = Int64.add and ( * ) = Int64.mul in
  let rec each lo hi f =
    if Int64.compare lo hi < 0 then (
      f lo;
      each (Int64.succ lo) hi f)
  in
  each 0L t.batch (fun bb ->
      each 0L t.m (fun i ->
          each 0L t.n (fun j ->
              let at =
                (bb * t.c.Output.batch) + (i * t.c.Output.row)
                + (j * t.c.Output.col)
              in
              let acc =
                ref (match t.seed with Seed.Zero -> 0. | Seed.Tensor -> c at)
              in
              each 0L t.k (fun kk ->
                  let x =
                    a
                      ((bb * t.a.Operand.batch) + (i * t.a.Operand.outer)
                     + (kk * t.a.Operand.k))
                  and y =
                    b
                      ((bb * t.b.Operand.batch) + (j * t.b.Operand.outer)
                     + (kk * t.b.Operand.k))
                  in
                  acc :=
                    match t.numerics with
                    | Numerics.Fused_sequential -> F32_math.fma32 x y !acc
                    | Numerics.Ordered -> round32 (!acc +. round32 (x *. y)));
              write at !acc)))

module Select = struct
  module Engine = struct
    type t = Loops | Rvv | Zvt

    let name = function Loops -> "loops" | Rvv -> "rvv" | Zvt -> "zvt"
  end

  module Reason = struct
    type t =
      | Format_off_matrix
      | Fused_off_matrix
      | No_vector_unit
      | Ordered_on_matrix
      | Vector_unit

    let name = function
      | Format_off_matrix -> "format_off_matrix"
      | Fused_off_matrix -> "fused_off_matrix"
      | No_vector_unit -> "no_vector_unit"
      | Ordered_on_matrix -> "ordered_on_matrix"
      | Vector_unit -> "vector_unit"
  end

  module Capabilities = struct
    type t = { matrix : bool; vector : bool }
  end

  let choose (c : Capabilities.t) (d : t) =
    match (c.matrix, d.format, d.numerics, c.vector) with
    | true, Element.F32, Numerics.Ordered, _ ->
        (Engine.Zvt, Reason.Ordered_on_matrix)
    | _, _, _, false -> (Engine.Loops, Reason.No_vector_unit)
    | true, Element.Bf16, _, true -> (Engine.Rvv, Reason.Format_off_matrix)
    | true, Element.F32, Numerics.Fused_sequential, true ->
        (Engine.Rvv, Reason.Fused_off_matrix)
    | false, _, _, true -> (Engine.Rvv, Reason.Vector_unit)
end

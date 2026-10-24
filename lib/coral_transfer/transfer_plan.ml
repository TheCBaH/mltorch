module C = Contraction

module A = struct
  type t = Copied | In_place | Streamed of { rows : int }
end

module B = struct
  type t = Copied | In_place | Shared
end

module Out = struct
  type t = Core | Dma
end

type t = { a : A.t; b : B.t; out : Out.t }

let in_place = { a = A.In_place; b = B.In_place; out = Out.Core }

let pp ppf { a; b; out } =
  Fmt.pf ppf "A %s, B %s%s"
    (match a with
    | A.Copied -> "copied"
    | A.In_place -> "in place"
    | A.Streamed { rows } -> Printf.sprintf "streamed by %d rows" rows)
    (match b with
    | B.Copied -> "copied"
    | B.In_place -> "in place"
    | B.Shared -> "shared")
    (match out with Out.Core -> "" | Out.Dma -> ", C by the DMA")

module Refusal = struct
  type t =
    | Block_rows of { rows : int; unit : int }
    | C_not_beats
    | C_not_streamed
    | C_rows_not_contiguous
    | Over_budget of { needed : int; budget : int }
    | Over_transfer of { bytes : int }
    | Rows_not_contiguous

  let pp ppf = function
    | Block_rows { rows; unit } ->
        Fmt.pf ppf "a block of %d rows, not a multiple of %d" rows unit
    | C_not_beats -> Fmt.pf ppf "C by the DMA, C's rows not whole beats"
    | C_not_streamed -> Fmt.pf ppf "C by the DMA, A not streamed"
    | C_rows_not_contiguous -> Fmt.pf ppf "C's rows are not contiguous"
    | Over_budget { needed; budget } ->
        Fmt.pf ppf "%d bytes of DTCM needed, %d given" needed budget
    | Over_transfer { bytes } ->
        Fmt.pf ppf "a copy of %d bytes, past one DMA transfer" bytes
    | Rows_not_contiguous -> Fmt.pf ppf "A's rows are not contiguous"
end

let size (d : C.t) = Int64.to_int (C.Element.bytes d.C.format)

let span (d : C.t) which =
  let k = Int64.max 1L d.C.k in
  let o, rows =
    match which with `A -> (d.C.a, d.C.m) | `B -> (d.C.b, d.C.n)
  in
  let last count stride = Int64.mul (Int64.max 0L (Int64.pred count)) stride in
  let elements =
    Int64.succ
      (List.fold_left Int64.add 0L
         [
           last d.C.batch o.C.Operand.batch;
           last rows o.C.Operand.outer;
           last k o.C.Operand.k;
         ])
  in
  ((Int64.to_int elements * size d) + 15) land lnot 15

let group_rows (d : C.t) =
  if C.Element.integer d.C.format then 3
  else
    match d.C.numerics with
    | C.Numerics.Fused_sequential -> 3
    | C.Numerics.Ordered -> 2

let row_bytes (d : C.t) = Int64.to_int d.C.k * size d
let rec gcd a b = if b = 0 then a else gcd b (a mod b)

(* One row of C, [n] binary32 or 32-bit integer results. *)
let c_row_bytes (d : C.t) = Int64.to_int d.C.n * 4

let unit_rows ?(out = Out.Core) d =
  let lcm a b = a * b / gcd a b in
  let beats bytes = 16 / gcd (max 1 bytes) 16 in
  lcm (group_rows d)
    (lcm
       (beats (row_bytes d))
       (match out with Out.Core -> 1 | Out.Dma -> beats (c_row_bytes d)))

let c_contiguous (d : C.t) =
  Int64.equal d.C.c.C.Output.col 1L && Int64.equal d.C.c.C.Output.row d.C.n

let rows_contiguous (d : C.t) =
  Int64.equal d.C.a.C.Operand.k 1L
  && Int64.equal d.C.a.C.Operand.outer d.C.k
  && Int64.compare d.C.k 0L > 0

(* A ring half: a block of A's rows, then, with C by the DMA, the block's
   rows of C *)
let half_bytes d ~rows ~out =
  rows
  * (row_bytes d + match out with Out.Core -> 0 | Out.Dma -> c_row_bytes d)

let bytes d { a; b; out } =
  (match a with
    | A.Copied -> span d `A
    | A.In_place -> 0
    | A.Streamed { rows } -> 2 * half_bytes d ~rows ~out)
  + match b with B.Copied | B.Shared -> span d `B | B.In_place -> 0

module Witness = struct
  type nonrec t = { plan : t; contraction : C.t; bytes : int }

  let plan w = w.plan
  let contraction w = w.contraction
  let bytes w = w.bytes
end

let max_transfer = 0xFF_FFF0

let check ~budget d p =
  let ( let* ) = Result.bind in
  let transfer n =
    if n > max_transfer then Error (Refusal.Over_transfer { bytes = n })
    else Ok ()
  in
  let* () =
    match p.a with
    | A.In_place -> Ok ()
    | A.Copied -> transfer (span d `A)
    | A.Streamed { rows } ->
        let unit = unit_rows ~out:p.out d in
        if not (rows_contiguous d) then Error Refusal.Rows_not_contiguous
        else if rows <= 0 || rows mod unit <> 0 then
          Error (Refusal.Block_rows { rows; unit })
        else transfer (rows * row_bytes d)
  in
  let* () =
    match p.b with
    | B.In_place -> Ok ()
    | B.Copied | B.Shared -> transfer (span d `B)
  in
  let* () =
    match (p.out, p.a) with
    | Out.Core, _ -> Ok ()
    | Out.Dma, (A.Copied | A.In_place) -> Error Refusal.C_not_streamed
    | Out.Dma, A.Streamed _ ->
        if not (c_contiguous d) then Error Refusal.C_rows_not_contiguous
        else if Int64.to_int d.C.m * c_row_bytes d mod 16 <> 0 then
          Error Refusal.C_not_beats
        else Ok ()
  in
  let needed = bytes d p in
  if needed > budget then Error (Refusal.Over_budget { needed; budget })
  else Ok { Witness.plan = p; contraction = d; bytes = needed }

let candidates ~budget d =
  let streamed b out =
    let left = budget - bytes d { a = A.In_place; b; out = Out.Core } in
    let unit = unit_rows ~out d in
    let half = max 1 (half_bytes d ~rows:1 ~out) in
    let up x = (x + unit - 1) / unit * unit in
    let rows =
      min (left / (2 * half) / unit * unit) (up (Int64.to_int d.C.m))
    in
    if rows_contiguous d && rows >= unit then [ (A.Streamed { rows }, out) ]
    else []
  in
  List.concat_map
    (fun b ->
      List.filter_map
        (fun (a, out) ->
          let p = { a; b; out } in
          match check ~budget d p with Ok _ -> Some p | Error _ -> None)
        ([ (A.In_place, Out.Core); (A.Copied, Out.Core) ]
        @ streamed b Out.Core @ streamed b Out.Dma))
    [ B.In_place; B.Copied; B.Shared ]

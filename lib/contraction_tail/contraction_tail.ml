module C = Contraction

module Side = struct
  type t = { name : string; batch : int64; row : int64; col : int64 }
end

module Tap = struct
  type t = { name : string; strides : C.Output.t }
end

module Op = struct
  type t =
    | Add of Side.t
    | Clamp of { lo : float; hi : float }
    | Mul of Side.t
    | Relu
end

module Step = struct
  type t = { op : Op.t; tap : Tap.t option }
end

type t = { core : C.t; steps : Step.t list }

module Invalid = struct
  type t =
    | Clamp_bounds of { lo : float; hi : float }
    | Duplicate_tap of string
    | Integer_core of C.Element.t
    | Stride of { name : string; value : int64 }
end

let make core steps =
  let open Err.Syntax in
  let fail i = Err.fail (`Invalid_contraction_tail i) in
  let stride name value =
    if Int64.compare value 0L < 0 then fail (Invalid.Stride { name; value })
    else Err.return ()
  in
  let side (s : Side.t) =
    let* () = stride (s.name ^ ".batch") s.batch in
    let* () = stride (s.name ^ ".row") s.row in
    stride (s.name ^ ".col") s.col
  in
  let rec steps_ok seen = function
    | [] -> Err.return ()
    | (st : Step.t) :: rest -> (
        let* () =
          match st.op with
          | Op.Add s | Op.Mul s -> side s
          | Op.Clamp { lo; hi } ->
              if Float.is_nan lo || Float.is_nan hi || lo > hi then
                fail (Invalid.Clamp_bounds { lo; hi })
              else Err.return ()
          | Op.Relu -> Err.return ()
        in
        match st.tap with
        | None -> steps_ok seen rest
        | Some tp ->
            if List.mem tp.name seen then fail (Invalid.Duplicate_tap tp.name)
            else
              let s = tp.strides in
              let* () = stride (tp.name ^ ".batch") s.C.Output.batch in
              let* () = stride (tp.name ^ ".row") s.C.Output.row in
              let* () = stride (tp.name ^ ".col") s.C.Output.col in
              steps_ok (tp.name :: seen) rest)
  in
  let* () =
    if C.Element.integer core.C.format then
      fail (Invalid.Integer_core core.C.format)
    else Err.return ()
  in
  let* () = steps_ok [] steps in
  Err.return { core; steps }

let pp ppf t =
  let op ppf = function
    | Op.Add s -> Fmt.pf ppf "add %s" s.Side.name
    | Op.Clamp { lo; hi } -> Fmt.pf ppf "clamp %h %h" lo hi
    | Op.Mul s -> Fmt.pf ppf "mul %s" s.Side.name
    | Op.Relu -> Fmt.string ppf "relu"
  in
  let step ppf (s : Step.t) =
    match s.tap with
    | None -> op ppf s.op
    | Some tp -> Fmt.pf ppf "%a -> %s" op s.op tp.Tap.name
  in
  Fmt.pf ppf "@[<h>%a; %a@]" C.pp t.core Fmt.(list ~sep:(any "; ") step) t.steps

let pp_error ppf (`Invalid_contraction_tail i) =
  match i with
  | Invalid.Clamp_bounds { lo; hi } ->
      Fmt.pf ppf "contraction tail clamp bounds %h, %h" lo hi
  | Invalid.Duplicate_tap n -> Fmt.pf ppf "contraction tail tap %s twice" n
  | Invalid.Integer_core f ->
      Fmt.pf ppf "contraction tail over %s, an integer product"
        (C.Element.name f)
  | Invalid.Stride { name; value } ->
      Fmt.pf ppf "contraction tail stride %s = %Ld" name value

module Block = struct
  type t = { rows : int64; cols : int64 }

  module Invalid = struct
    type t =
      | Extent of { name : string; value : int64 }
      | Too_large of { rows : int64; cols : int64; limit : int64 }
  end

  let limit = 1_048_576L

  let make ~rows ~cols =
    let fail i = Err.fail (`Invalid_block i) in
    if Int64.compare rows 1L < 0 then
      fail (Invalid.Extent { name = "rows"; value = rows })
    else if Int64.compare cols 1L < 0 then
      fail (Invalid.Extent { name = "cols"; value = cols })
    else if Int64.compare cols (Int64.div limit rows) > 0 then
      (* a division, so the product is never formed past the limit *)
      fail (Invalid.Too_large { rows; cols; limit })
    else Err.return { rows; cols }

  let cells t = Int64.mul t.rows t.cols

  let pp_error ppf (`Invalid_block i) =
    match i with
    | Invalid.Extent { name; value } ->
        Fmt.pf ppf "block extent %s = %Ld" name value
    | Invalid.Too_large { rows; cols; limit } ->
        Fmt.pf ppf "block %Ld x %Ld past %Ld cells" rows cols limit
end

module Fault = struct
  type t = Seed_from_scratch | Side_block_local | Unrounded_intermediate
end

let ( + ) = Int64.add
let ( * ) = Int64.mul

let rec each lo hi f =
  if Int64.compare lo hi < 0 then (
    f lo;
    each (Int64.succ lo) hi f)

(* [f start len] over consecutive pieces of at most [size] covering [0, total) *)
let rec pieces ~total ~size ~from f =
  if Int64.compare from total < 0 then (
    let len = Int64.min size (Int64.sub total from) in
    f from len;
    pieces ~total ~size ~from:(from + len) f)

(* The steps over one output value at coordinates [(b, i, j)]; side operands
   are read at [(si, sj)]. *)
let tail ~round ~side ~tap t ~b ~i ~j ~si ~sj v =
  let read (s : Side.t) =
    side s.name ((b * s.batch) + (si * s.row) + (sj * s.col))
  in
  List.fold_left
    (fun v (st : Step.t) ->
      let v =
        match st.op with
        | Op.Add s -> round (v +. read s)
        | Op.Mul s -> round (v *. read s)
        | Op.Clamp { lo; hi } -> if v < lo then lo else if v > hi then hi else v
        | Op.Relu -> if v < 0. then 0. else v
      in
      (match st.tap with
      | None -> ()
      | Some tp ->
          let s = tp.Tap.strides in
          tap tp.Tap.name
            ((b * s.C.Output.batch) + (i * s.C.Output.row) + (j * s.C.Output.col)
            )
            v);
      v)
    v t.steps

let c_offset (o : C.Output.t) ~b ~i ~j =
  (b * o.batch) + (i * o.row) + (j * o.col)

let eval_reference t ~a ~b ~c ~side ~tap ~write =
  let core = t.core in
  let whole = Hashtbl.create 64 in
  C.eval core ~a ~b ~c ~write:(fun off v -> Hashtbl.replace whole off v);
  each 0L core.C.batch (fun bb ->
      each 0L core.C.m (fun i ->
          each 0L core.C.n (fun j ->
              let off = c_offset core.C.c ~b:bb ~i ~j in
              write off
                (tail ~round:F32_math.round32 ~side ~tap t ~b:bb ~i ~j ~si:i
                   ~sj:j (Hashtbl.find whole off)))))

let eval_scratch ?fault t (blk : Block.t) ~a ~b ~c ~side ~tap ~write =
  let core = t.core in
  let o = core.C.c in
  let round =
    match fault with
    | Some Fault.Unrounded_intermediate -> Fun.id
    | _ -> F32_math.round32
  in
  each 0L core.C.batch (fun bb ->
      pieces ~total:core.C.m ~size:blk.Block.rows ~from:0L (fun i0 bm ->
          pieces ~total:core.C.n ~size:blk.Block.cols ~from:0L (fun j0 bn ->
              let a_base =
                (bb * core.C.a.C.Operand.batch) + (i0 * core.C.a.outer)
              and b_base =
                (bb * core.C.b.C.Operand.batch) + (j0 * core.C.b.outer)
              in
              (* a scratch offset, back to the output's own *)
              let c_at s =
                c_offset o ~b:bb
                  ~i:(i0 + Int64.div s bn)
                  ~j:(j0 + Int64.rem s bn)
              in
              let sub =
                match
                  Err.payload
                    (C.make ~batch:1L ~m:bm ~n:bn ~k:core.C.k
                       ~a:{ core.C.a with C.Operand.batch = 0L }
                       ~b:{ core.C.b with C.Operand.batch = 0L }
                       ~c:{ C.Output.batch = bm * bn; row = bn; col = 1L }
                       ~format:core.C.format ~seed:core.C.seed
                       ~numerics:core.C.numerics)
                with
                | Ok s -> s
                | Error _ ->
                    invalid_arg "Contraction_tail: a block of a legal core"
              in
              (* [Block.make] bounds [rows * cols], so cells index an array *)
              let scratch = Array.make (Int64.to_int (bm * bn)) 0. in
              let seed =
                match fault with
                | Some Fault.Seed_from_scratch ->
                    fun s -> scratch.(Int64.to_int s)
                | _ -> fun s -> c (c_at s)
              in
              C.eval sub
                ~a:(fun off -> a (a_base + off))
                ~b:(fun off -> b (b_base + off))
                ~c:seed
                ~write:(fun s v -> scratch.(Int64.to_int s) <- v);
              each 0L bm (fun r ->
                  each 0L bn (fun q ->
                      let i = i0 + r and j = j0 + q in
                      let si, sj =
                        match fault with
                        | Some Fault.Side_block_local -> (r, q)
                        | _ -> (i, j)
                      in
                      let v = scratch.(Int64.to_int ((r * bn) + q)) in
                      write (c_offset o ~b:bb ~i ~j)
                        (tail ~round ~side ~tap t ~b:bb ~i ~j ~si ~sj v))))))

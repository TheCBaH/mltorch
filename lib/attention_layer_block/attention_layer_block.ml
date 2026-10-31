module A = Attention_block
module C = Contraction
module P = Attention_block.Precision

module Placement = struct
  type t = Post | Pre

  let name = function Post -> "post" | Pre -> "pre"
end

module Norm_spec = struct
  type t = {
    kind : Norm_block.Kind.t;
    placement : Placement.t;
    eps : float;
    weight : bool;
    bias : bool;
  }
end

module Norm = struct
  type t = { block : Norm_block.t; placement : Placement.t }
end

module Rope = struct
  type t = { batch : int64; row : int64; col : int64 }
end

module Flags = struct
  type t = {
    q_bias : bool;
    k_bias : bool;
    v_bias : bool;
    o_bias : bool;
    layer_scale : bool;
  }

  let none =
    {
      q_bias = false;
      k_bias = false;
      v_bias = false;
      o_bias = false;
      layer_scale = false;
    }

  let all =
    {
      q_bias = true;
      k_bias = true;
      v_bias = true;
      o_bias = true;
      layer_scale = true;
    }
end

type t = {
  batch : int64;
  tokens : int64;
  width : int64;
  heads : int64;
  d : int64;
  norm : Norm.t option;
  q : C.t;
  k : C.t;
  v : C.t;
  o : C.t;
  flags : Flags.t;
  rope : Rope.t option;
  attend : A.t;
  precision : P.t;
}

module Invalid = struct
  type t =
    | Attention of A.Invalid.t
    | Contraction of C.Invalid.t
    | Extent of { name : string; value : int64 }
    | Norm of Norm_block.Invalid.t
    | Rope_odd of int64
    | Rope_stride of { name : string; value : int64 }
    | Width of { width : int64; heads : int64 }
end

let fail i = Err.fail (`Invalid_attention_layer i)

let make ~batch ~tokens ~width ~heads ?norm ~flags ?rope ?mask ~scale
    ~block_keys ~precision () =
  let open Err.Syntax in
  let extent name value =
    if Int64.compare value 1L < 0 then fail (Invalid.Extent { name; value })
    else Err.return ()
  in
  let* () = extent "batch" batch in
  let* () = extent "tokens" tokens in
  let* () = extent "width" width in
  let* () = extent "heads" heads in
  let* () =
    if Int64.rem width heads <> 0L then fail (Invalid.Width { width; heads })
    else Err.return ()
  in
  let d = Int64.div width heads in
  let* () =
    match rope with
    | None -> Err.return ()
    | Some (r : Rope.t) ->
        let one name value =
          if Int64.compare value 0L < 0 then
            fail (Invalid.Rope_stride { name; value })
          else Err.return ()
        in
        let* () = one "batch" r.Rope.batch in
        let* () = one "row" r.Rope.row in
        let* () = one "col" r.Rope.col in
        if Int64.rem d 2L <> 0L then fail (Invalid.Rope_odd d)
        else Err.return ()
  in
  let rows = Int64.mul batch tokens in
  let dense = { Norm_block.Strides.row = width; col = 1L } in
  let* norm =
    match norm with
    | None -> Err.return None
    | Some (s : Norm_spec.t) -> (
        (* a bfloat16 layer rounds a Pre norm's output, an operand of the
           paired products; a Post norm's output is the layer's, binary32 *)
        let precision =
          match (precision, s.Norm_spec.placement) with
          | P.Bf16_paired, Placement.Post -> P.F32_ordered
          | (P.Bf16_paired | P.F32_fused | P.F32_ordered), _ -> precision
        in
        match
          Err.payload
            (Norm_block.make ~kind:s.Norm_spec.kind ~rows ~width
               ~eps:s.Norm_spec.eps ~weight:s.Norm_spec.weight
               ~bias:s.Norm_spec.bias ~x:dense ~y:dense ~precision)
        with
        | Ok block ->
            Err.return (Some { Norm.block; placement = s.Norm_spec.placement })
        | Error (`Invalid_norm i) -> fail (Invalid.Norm i))
  in
  let projection =
    match
      Err.payload
        (C.make ~batch:1L ~m:rows ~n:width ~k:width
           ~a:{ C.Operand.batch = 0L; outer = width; k = 1L }
           ~b:{ C.Operand.batch = 0L; outer = width; k = 1L }
           ~c:{ C.Output.batch = 0L; row = width; col = 1L }
           ~format:(P.format precision) ~seed:C.Seed.Zero
           ~numerics:(P.numerics precision))
    with
    | Ok c -> Err.return c
    | Error (`Invalid_contraction i) -> fail (Invalid.Contraction i)
  in
  let* q = projection in
  let heads_strides =
    {
      A.Strides.batch = Int64.mul tokens width;
      head = d;
      row = width;
      col = 1L;
    }
  in
  let* attend =
    match
      Err.payload
        (A.make ~batch ~heads ~tq:tokens ~tk:tokens ~d ~q:heads_strides
           ~k:heads_strides ~v:heads_strides ~o:heads_strides ?mask ~scale
           ~block_keys ~precision ())
    with
    | Ok a -> Err.return a
    | Error (`Invalid_attention i) -> fail (Invalid.Attention i)
  in
  Err.return
    {
      batch;
      tokens;
      width;
      heads;
      d;
      norm;
      q;
      k = q;
      v = q;
      o = q;
      flags;
      rope;
      attend;
      precision;
    }

let pp_flags ppf (f : Flags.t) =
  let on name b = if b then [ name ] else [] in
  Fmt.(list ~sep:(any ",") string)
    ppf
    (on "q_bias" f.Flags.q_bias @ on "k_bias" f.Flags.k_bias
   @ on "v_bias" f.Flags.v_bias @ on "o_bias" f.Flags.o_bias
    @ on "layer_scale" f.Flags.layer_scale)

let pp ppf t =
  Fmt.pf ppf
    "@[<h>attention_layer batch=%Ld tokens=%Ld width=%Ld heads=%Ld norm=%a \
     flags=[%a] rope=%a %a@]"
    t.batch t.tokens t.width t.heads
    Fmt.(
      option ~none:(any "none") (fun ppf (n : Norm.t) ->
          Fmt.pf ppf "%s(%a)"
            (Placement.name n.Norm.placement)
            Norm_block.pp n.Norm.block))
    t.norm pp_flags t.flags
    Fmt.(
      option ~none:(any "none") (fun ppf (r : Rope.t) ->
          Fmt.pf ppf "(%Ld,%Ld,%Ld)" r.Rope.batch r.Rope.row r.Rope.col))
    t.rope A.pp t.attend

let pp_error ppf (`Invalid_attention_layer i) =
  match i with
  | Invalid.Attention a -> A.pp_error ppf (`Invalid_attention a)
  | Invalid.Contraction c -> C.pp_error ppf (`Invalid_contraction c)
  | Invalid.Extent { name; value } ->
      Fmt.pf ppf "attention layer extent %s = %Ld" name value
  | Invalid.Norm n -> Norm_block.pp_error ppf (`Invalid_norm n)
  | Invalid.Rope_odd d ->
      Fmt.pf ppf "a rotary embedding over an odd head dimension %Ld" d
  | Invalid.Rope_stride { name; value } ->
      Fmt.pf ppf "rotary stride %s = %Ld" name value
  | Invalid.Width { width; heads } ->
      Fmt.pf ppf "attention layer width %Ld is not a multiple of %Ld heads"
        width heads

(* The name: the layer's own fields, then the attention's scale, mask and
   block, from which [make] derives every part again. *)
let name t =
  let flags (f : Flags.t) =
    let on c b = if b then c else "" in
    on "q" f.Flags.q_bias ^ on "k" f.Flags.k_bias ^ on "v" f.Flags.v_bias
    ^ on "o" f.Flags.o_bias ^ on "g" f.Flags.layer_scale
  in
  let norm =
    match t.norm with
    | None -> "nonorm"
    | Some n ->
        let b = n.Norm.block in
        Printf.sprintf "n%s_%s_%lx_%s%s"
          (Norm_block.Kind.name b.Norm_block.kind)
          (Placement.name n.Norm.placement)
          (Int32.bits_of_float b.Norm_block.eps)
          (if b.Norm_block.weight then "w" else "")
          (if b.Norm_block.bias then "b" else "")
  in
  let rope =
    match t.rope with
    | None -> "norope"
    | Some r -> Printf.sprintf "r%Ld_%Ld_%Ld" r.Rope.batch r.Rope.row r.Rope.col
  in
  let mask =
    match t.attend.A.mask with
    | None -> "nomask"
    | Some m ->
        Printf.sprintf "m%Ld_%Ld_%Ld_%Ld" m.A.Strides.batch m.A.Strides.head
          m.A.Strides.row m.A.Strides.col
  in
  Printf.sprintf
    "attention_layer.b%Ld.t%Ld.e%Ld.h%Ld.%s.f%s.%s.%s.x%lx.n%Ld.p%s" t.batch
    t.tokens t.width t.heads norm (flags t.flags) rope mask
    (Int32.bits_of_float t.attend.A.scale)
    t.attend.A.block_keys (P.name t.precision)

let of_name s =
  let tagged tag field =
    let n = String.length tag in
    if String.length field >= n && String.sub field 0 n = tag then
      Some (String.sub field n (String.length field - n))
    else None
  in
  let int_of tag field = Option.bind (tagged tag field) Int64.of_string_opt in
  let ints tag field =
    Option.bind (tagged tag field) (fun text ->
        let parts =
          List.map Int64.of_string_opt (String.split_on_char '_' text)
        in
        if List.for_all Option.is_some parts then
          Some (List.map Option.get parts)
        else None)
  in
  let hex text = Int32.of_string_opt ("0x" ^ text) in
  let norm field =
    if field = "nonorm" then Some None
    else
      match Option.map (String.split_on_char '_') (tagged "n" field) with
      | Some [ kind; placement; eps; affine ] -> (
          let kind =
            List.find_opt
              (fun k -> Norm_block.Kind.name k = kind)
              [ Norm_block.Kind.Layer; Norm_block.Kind.Rms ]
          and placement =
            List.find_opt
              (fun p -> Placement.name p = placement)
              [ Placement.Post; Placement.Pre ]
          and affine =
            match affine with
            | "" -> Some (false, false)
            | "w" -> Some (true, false)
            | "b" -> Some (false, true)
            | "wb" -> Some (true, true)
            | _ -> None
          in
          match (kind, placement, hex eps, affine) with
          | Some kind, Some placement, Some eps, Some (weight, bias) ->
              Some
                (Some
                   {
                     Norm_spec.kind;
                     placement;
                     eps = Int32.float_of_bits eps;
                     weight;
                     bias;
                   })
          | _ -> None)
      | _ -> None
  in
  let flags field =
    Option.bind (tagged "f" field) (fun text ->
        let has c = String.contains text c in
        if
          String.for_all (fun c -> String.contains "qkvog" c) text
          && String.length text
             = List.length (List.filter has [ 'q'; 'k'; 'v'; 'o'; 'g' ])
        then
          Some
            {
              Flags.q_bias = has 'q';
              k_bias = has 'k';
              v_bias = has 'v';
              o_bias = has 'o';
              layer_scale = has 'g';
            }
        else None)
  in
  let rope field =
    if field = "norope" then Some None
    else
      match ints "r" field with
      | Some [ batch; row; col ] -> Some (Some { Rope.batch; row; col })
      | _ -> None
  in
  let mask field =
    if field = "nomask" then Some None
    else
      match ints "m" field with
      | Some [ batch; head; row; col ] ->
          Some (Some { A.Strides.batch; head; row; col })
      | _ -> None
  in
  match String.split_on_char '.' s with
  | [ "attention_layer"; b; tk; e; h; n; f; r; m; x; bk; p ] -> (
      match
        ( int_of "b" b,
          int_of "t" tk,
          int_of "e" e,
          int_of "h" h,
          norm n,
          flags f,
          rope r,
          mask m,
          Option.bind (tagged "x" x) hex,
          int_of "n" bk,
          Option.bind (tagged "p" p) (fun text ->
              List.find_opt (fun p -> P.name p = text) P.all) )
      with
      | ( Some batch,
          Some tokens,
          Some width,
          Some heads,
          Some norm,
          Some flags,
          Some rope,
          Some mask,
          Some scale,
          Some block_keys,
          Some precision ) ->
          Result.to_option
            (Err.payload
               (make ~batch ~tokens ~width ~heads ?norm ~flags ?rope ?mask
                  ~scale:(Int32.float_of_bits scale)
                  ~block_keys ~precision ()))
      | _ -> None)
  | _ -> None

let symbol t =
  "mlt_attention_layer_"
  ^ String.sub (Digest.to_hex (Digest.string (name t))) 0 16

module Readers = struct
  type t = {
    x : int64 -> float;
    norm_weight : int64 -> float;
    norm_bias : int64 -> float;
    wq : int64 -> float;
    bq : int64 -> float;
    wk : int64 -> float;
    bk : int64 -> float;
    wv : int64 -> float;
    bv : int64 -> float;
    wo : int64 -> float;
    bo : int64 -> float;
    gamma : int64 -> float;
    cos : int64 -> float;
    sin : int64 -> float;
    mask : int64 -> float;
  }
end

let nan32 = Int32.float_of_bits 0x7FC00000l
let r32 x = if Float.is_nan x then nan32 else F32_math.round32 x

let eval t (rd : Readers.t) ~write =
  (* sizes are bounded by the caller's operands; the descriptor's extents are
     [int64] and narrow here, where the arrays are allocated *)
  let int = Int64.to_int in
  let rows = int (Int64.mul t.batch t.tokens) and e = int t.width in
  let cells = rows * e in
  let at a o = a.(int o) in
  let into a o v = a.(int o) <- v in
  let bf16 = t.precision = P.Bf16_paired in
  let operand v = if bf16 then A.bf16_round v else v in
  let pre, post =
    match t.norm with
    | None -> (None, None)
    | Some { Norm.block; placement = Placement.Pre } -> (Some block, None)
    | Some { Norm.block; placement = Placement.Post } -> (None, Some block)
  in
  let x = Array.init cells (fun i -> rd.Readers.x (Int64.of_int i)) in
  let u =
    match pre with
    | Some block ->
        let u = Array.make cells 0. in
        Norm_block.eval block ~x:(at x) ~weight:rd.Readers.norm_weight
          ~bias:rd.Readers.norm_bias ~write:(into u);
        u
    | None -> Array.map operand x
  in
  let project (c : C.t) ~a ~w ~bias ~biased =
    let out = Array.make cells 0. in
    C.eval c ~a:(at a) ~b:w ~c:(fun _ -> 0.) ~write:(into out);
    if biased then
      Array.iteri
        (fun i v -> out.(i) <- r32 (v +. bias (Int64.of_int (i mod e))))
        out;
    out
  in
  let f = t.flags in
  let q =
    project t.q ~a:u ~w:rd.Readers.wq ~bias:rd.Readers.bq ~biased:f.Flags.q_bias
  and k =
    project t.k ~a:u ~w:rd.Readers.wk ~bias:rd.Readers.bk ~biased:f.Flags.k_bias
  and v =
    project t.v ~a:u ~w:rd.Readers.wv ~bias:rd.Readers.bv ~biased:f.Flags.v_bias
  in
  (match t.rope with
  | None -> ()
  | Some r ->
      let d = int t.d and half = int t.d / 2 and l = int t.tokens in
      let rotate z =
        let src = Array.copy z in
        for i = 0 to cells - 1 do
          let row = i / e and col = i mod e in
          let b = row / l and tok = row mod l and f = col mod d in
          let table read =
            read
              (Int64.add
                 (Int64.mul (Int64.of_int b) r.Rope.batch)
                 (Int64.add
                    (Int64.mul (Int64.of_int tok) r.Rope.row)
                    (Int64.mul (Int64.of_int f) r.Rope.col)))
          in
          let partner =
            if f < half then r32 (src.(i + half) *. -1.) else src.(i - half)
          in
          z.(i) <-
            r32
              (r32 (src.(i) *. table rd.Readers.cos)
              +. r32 (partner *. table rd.Readers.sin))
        done
      in
      rotate q;
      rotate k);
  let q = Array.map operand q
  and k = Array.map operand k
  and v = Array.map operand v in
  let a = Array.make cells 0. in
  A.eval t.attend ~q:(at q) ~k:(at k) ~v:(at v) ~mask:rd.Readers.mask
    ~write:(into a);
  let a = Array.map operand a in
  let o =
    project t.o ~a ~w:rd.Readers.wo ~bias:rd.Readers.bo ~biased:f.Flags.o_bias
  in
  let res =
    Array.mapi
      (fun i v ->
        let v =
          if f.Flags.layer_scale then
            r32 (v *. rd.Readers.gamma (Int64.of_int (i mod e)))
          else v
        in
        r32 (v +. x.(i)))
      o
  in
  match post with
  | Some block ->
      Norm_block.eval block ~x:(at res) ~weight:rd.Readers.norm_weight
        ~bias:rd.Readers.norm_bias ~write
  | None -> Array.iteri (fun i v -> write (Int64.of_int i) v) res

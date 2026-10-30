open Graph_ir
module A = Attention_block

module Grant = struct
  type t = { block_keys : int }

  let default = { block_keys = 32 }
end

module Reason = struct
  type t =
    | Block_keys of int
    | Descriptor of string
    | Engine_absent
    | Feature_extent of int
    | Mask_not_f32
    | Not_bf16 of string
    | Not_sdpa
    | Outer_axes
    | Padded_work of { padded : int64; actual : int64 }
    | Ungranted
    | Unknown_node

  let pp ppf = function
    | Block_keys n ->
        Fmt.pf ppf "a block of %d keys, neither at most 16 nor 32" n
    | Descriptor why -> Fmt.pf ppf "the descriptor: %s" why
    | Engine_absent -> Fmt.string ppf "the core has no matrix engine"
    | Feature_extent d -> Fmt.pf ppf "a head dimension of %d, odd or past 64" d
    | Mask_not_f32 -> Fmt.string ppf "a mask that is not binary32"
    | Not_bf16 role -> Fmt.pf ppf "the %s is not bfloat16" role
    | Not_sdpa -> Fmt.string ppf "not an attention"
    | Outer_axes -> Fmt.string ppf "an extent past 1 on the N or T axis"
    | Padded_work { padded; actual } ->
        Fmt.pf ppf "%Ld products for %Ld, more than twice the work" padded
          actual
    | Ungranted -> Fmt.string ppf "bf16_attention is not granted"
    | Unknown_node -> Fmt.string ppf "no such node"
end

let extent shape axis = Dim.to_int (Vec6.get shape axis)

(* Row-major element strides of a dense tensor, 0 on an axis of extent 1 (a
   broadcast axis reads its one element). *)
let stride shape axis =
  let axes = [ Axis.N; Axis.T; Axis.D; Axis.H; Axis.W; Axis.C ] in
  let rec after = function
    | [] -> []
    | a :: rest -> if a = axis then rest else after rest
  in
  if extent shape axis = 1 then 0L
  else
    Int64.of_int (List.fold_left (fun n a -> n * extent shape a) 1 (after axes))

let strides shape ~row ~col : A.Strides.t =
  {
    A.Strides.batch = stride shape Axis.D;
    head = stride shape Axis.H;
    row = stride shape row;
    col = stride shape col;
  }

let is_bf16 (sg : Tensor_sig.t) =
  match sg.Tensor_sig.fmt with Payload.Fmt Payload.BF16 -> true | _ -> false

let is_f32 (sg : Tensor_sig.t) =
  match sg.Tensor_sig.fmt with Payload.Fmt Payload.F32 -> true | _ -> false

let choose ~grant (caps : Contraction.Select.Capabilities.t) (g : graph) id =
  let ( let* ) = Result.bind in
  let* grant = Option.to_result ~none:Reason.Ungranted grant in
  let* () =
    if caps.Contraction.Select.Capabilities.matrix then Ok ()
    else Error Reason.Engine_absent
  in
  let* node =
    Option.to_result ~none:Reason.Unknown_node
      (List.find_opt (fun (n : node) -> Node_id.equal n.Node.id id) (nodes g))
  in
  let* s =
    match node.Node.op with Sdpa s -> Ok s | _ -> Error Reason.Not_sdpa
  in
  let sig_of r = Tensor_id.Map.find r g.Graph.tensors in
  let q = sig_of s.Attention.Sdpa.query
  and k = sig_of s.Attention.Sdpa.key
  and v = sig_of s.Attention.Sdpa.value in
  let* () = if is_bf16 q then Ok () else Error (Reason.Not_bf16 "query") in
  let* () = if is_bf16 k then Ok () else Error (Reason.Not_bf16 "key") in
  let* () = if is_bf16 v then Ok () else Error (Reason.Not_bf16 "value") in
  let mask = Option.map sig_of s.Attention.Sdpa.mask in
  let* () =
    match mask with
    | Some m when not (is_f32 m) -> Error Reason.Mask_not_f32
    | _ -> Ok ()
  in
  let output =
    match node.Node.outputs with
    | [ o ] -> o
    | _ -> invalid_arg "Sdpa_select: an attention has one output"
  in
  let o = sig_of output in
  let shapes =
    [ q; k; v; o ] @ Option.to_list mask
    |> List.map (fun (sg : Tensor_sig.t) -> sg.Tensor_sig.shape)
  in
  let* () =
    if
      List.for_all
        (fun sh -> extent sh Axis.N = 1 && extent sh Axis.T = 1)
        shapes
    then Ok ()
    else Error Reason.Outer_axes
  in
  let qs = q.Tensor_sig.shape and ks = k.Tensor_sig.shape in
  let os = o.Tensor_sig.shape in
  let d = extent qs Axis.C in
  let* () =
    if d mod 2 = 0 && d <= 64 then Ok () else Error (Reason.Feature_extent d)
  in
  let bk = grant.Grant.block_keys in
  let* () =
    if bk > 0 && bk mod 4 = 0 && (bk <= 16 || bk = 32) then Ok ()
    else Error (Reason.Block_keys bk)
  in
  let tq = extent qs Axis.W and tk = extent ks Axis.W in
  let padded =
    let up a b = (a + b - 1) / b * b in
    Int64.mul (Int64.of_int (up tq 16)) (Int64.of_int (up tk bk))
  and actual = Int64.mul (Int64.of_int tq) (Int64.of_int tk) in
  let* () =
    if Int64.compare padded (Int64.mul 2L actual) <= 0 then Ok ()
    else Error (Reason.Padded_work { padded; actual })
  in
  let scale =
    match s.Attention.Sdpa.params.Attention.Sdpa.scale with
    | Attention.Sdpa.Scale.Default -> A.Scale.default ~d:(Int64.of_int d)
    | Attention.Sdpa.Scale.Explicit x -> A.Scale.explicit x
  in
  let i64 = Int64.of_int in
  let* block =
    Result.map_error
      (fun e -> Reason.Descriptor (Fmt.str "%a" A.pp_error e))
      (Err.payload
         (A.make
            ~batch:(i64 (extent os Axis.D))
            ~heads:(i64 (extent os Axis.H))
            ~tq:(i64 tq) ~tk:(i64 tk) ~d:(i64 d)
            ~q:(strides qs ~row:Axis.W ~col:Axis.C)
            ~k:(strides ks ~row:Axis.W ~col:Axis.C)
            ~v:(strides v.Tensor_sig.shape ~row:Axis.W ~col:Axis.C)
            ~o:(strides os ~row:Axis.W ~col:Axis.C)
            ?mask:
              (Option.map
                 (fun (m : Tensor_sig.t) ->
                   strides m.Tensor_sig.shape ~row:Axis.W ~col:Axis.C)
                 mask)
            ~scale ~block_keys:(i64 bk) ()))
  in
  Ok
    {
      Attention_site.block;
      query = s.Attention.Sdpa.query;
      key = s.Attention.Sdpa.key;
      value = s.Attention.Sdpa.value;
      mask = s.Attention.Sdpa.mask;
      output;
    }

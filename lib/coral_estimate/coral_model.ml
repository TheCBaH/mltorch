module Core = struct
  type t = { matrix : bool; vector : bool; packed : Contraction.t -> bool }

  let vme ~packed = { matrix = true; vector = true; packed }
end

let ceil_div a b = Int64.div (Int64.add a (Int64.pred b)) b
let f = Int64.to_float

let format_name (d : Contraction.t) =
  match d.Contraction.format with
  | Contraction.Element.Bf16 -> "bf16"
  | Contraction.Element.F32 -> "f32"
  | Contraction.Element.I8 _ -> "i8"

let rn = Coral_census.Region.name

let contract (core : Core.t) (call : Coral_census.Contract_call.t) w =
  let d = call.Coral_census.Contract_call.desc in
  let caps =
    {
      Contraction.Select.Capabilities.matrix = core.Core.matrix;
      vector = core.Core.vector;
    }
  in
  let engine =
    match call.Coral_census.Contract_call.engine with
    | Some e -> e
    | None -> fst (Contraction.Select.choose caps d)
  in
  match engine with
  | Contraction.Select.Engine.Zvt ->
      (* per 16 x 16 output tile: its setup and C's stores (by C's region),
         and each k step's loads of the tile's A rows and B columns (by A's
         and B's) *)
      let mt = ceil_div d.Contraction.m 16L
      and nt = ceil_div d.Contraction.n 16L in
      let b = d.Contraction.batch and k = d.Contraction.k in
      let tiles = Int64.mul b (Int64.mul mt nt) in
      let rows_k = Int64.mul b (Int64.mul (Int64.mul d.Contraction.m nt) k) in
      let cols_k = Int64.mul b (Int64.mul (Int64.mul d.Contraction.n mt) k) in
      let p =
        Printf.sprintf "zvt.%s%s" (format_name d)
          (if core.Core.packed d then ".packed" else "")
      in
      [
        (p ^ ".call", w);
        (p ^ ".tiles." ^ rn call.Coral_census.Contract_call.c, w *. f tiles);
        (p ^ ".rows_k." ^ rn call.Coral_census.Contract_call.a, w *. f rows_k);
        (p ^ ".cols_k." ^ rn call.Coral_census.Contract_call.b, w *. f cols_k);
      ]
  | Contraction.Select.Engine.Rvv ->
      [ ("rvv.call", w); ("rvv.macs", w *. f (Contraction.macs d)) ]
  | Contraction.Select.Engine.Loops ->
      [ ("loops.macs", w *. f (Contraction.macs d)) ]

let attend (call : Coral_census.Attend_call.t) w =
  let a = call.Coral_census.Attend_call.desc in
  let heads = Int64.mul a.Attention_block.batch a.Attention_block.heads in
  let qblocks = Int64.mul heads (ceil_div a.Attention_block.tq 16L) in
  let kb = ceil_div a.Attention_block.tk a.Attention_block.block_keys in
  let kblocks = Int64.mul qblocks kb in
  let scores = f kblocks *. 16. *. f a.Attention_block.block_keys in
  let q = rn call.Coral_census.Attend_call.q
  and kv = rn call.Coral_census.Attend_call.kv
  and o = rn call.Coral_census.Attend_call.o in
  [
    ("attend.call", w);
    ("attend.qblocks." ^ q, w *. f qblocks);
    ("attend.out." ^ o, w *. f qblocks *. f a.Attention_block.d);
    ("attend.kblocks." ^ kv, w *. f kblocks);
    ("attend.scores", w *. scores);
    ("attend.dwork." ^ kv, w *. scores *. f a.Attention_block.d);
    (* the rows staged into the DTCM before packing: each query block's 16
       rows of Q, each key block's rows of K and of V, d elements each *)
    ("attend.stage_q." ^ q, w *. f qblocks *. 16. *. f a.Attention_block.d);
    ( "attend.stage_kv." ^ kv,
      w *. f kblocks *. 2.
      *. f a.Attention_block.block_keys
      *. f a.Attention_block.d );
  ]

let merge l =
  List.fold_left
    (fun acc (k, v) ->
      if List.mem_assoc k acc then
        List.map
          (fun (k', v') -> if k' = k then (k', v' +. v) else (k', v'))
          acc
      else acc @ [ (k, v) ])
    [] l

(* The vector unit runs an operation while earlier loads and stores are still
   in flight, so in one loop body the shorter of the operations' cycles and
   the memory instructions' is partly hidden by the longer. *)
let vector_compute = function
  | Coral_census.Class.Vec_f32_div | Vec_f32_fma | Vec_f32_op | Vec_math _ ->
      true
  | _ -> false

(* Only the accesses outside the DTCM: the waveforms show the external loads
   absorbing about three fifths of a vector divide's cycles and the DTCM's
   about an eighth. *)
let vector_memory = function
  | Coral_census.Class.Vec_load r
  | Vec_load_strided r
  | Vec_store r
  | Vec_store_strided r ->
      r <> Coral_census.Region.Dtcm
  | _ -> false

let hidden rates (c : Coral_census.t) =
  let cycles pred body =
    List.fold_left
      (fun acc (cl, n) ->
        if pred cl then
          acc
          +. n
             *. Option.value
                  (List.assoc_opt ("class." ^ Coral_census.Class.name cl) rates)
                  ~default:0.
        else acc)
      0. body
  in
  List.fold_left
    (fun acc body ->
      acc +. Float.min (cycles vector_compute body) (cycles vector_memory body))
    0. c.Coral_census.bodies

let features ?(rates = []) core (c : Coral_census.t) =
  let h = hidden rates c in
  merge
    (("invocation", 1.)
     :: List.map
          (fun (cl, v) -> ("class." ^ Coral_census.Class.name cl, v))
          c.Coral_census.counts
    @ List.concat_map (fun (d, w) -> contract core d w) c.Coral_census.contracts
    @ List.concat_map (fun (a, w) -> attend a w) c.Coral_census.attends
    @ if h = 0. then [] else [ ("overlap", -.h) ])

module Rates = struct
  type t = (string * float) list

  let predict r fs =
    List.fold_left
      (fun acc (k, v) ->
        acc +. (v *. Option.value (List.assoc_opt k r) ~default:0.))
      0. fs
end

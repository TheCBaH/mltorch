(* What the passes that expand one operation into a builder-made program share:
   index arithmetic over constant strides, counted loops, f32 cell access in the
   binary64 form every lowered program has, and splicing the built body in place
   of the operation. *)

module B = Ssa_builder

(* The sum of [stride * index] terms, a stride of zero dropped. *)
let offset b terms =
  let term (stride, iv) =
    if Int64.equal stride 0L then None
    else if Int64.equal stride 1L then Some iv
    else Some (B.index_scale b stride iv)
  in
  match
    List.fold_left
      (fun acc t ->
        match (acc, term t) with
        | acc, None -> acc
        | None, t -> t
        | Some a, Some t -> Some (B.index_add b a t))
      None terms
  with
  | Some v -> v
  | None -> B.index b 0L

(* [for i in 0 .. n) body], nothing when [n] is not positive. *)
let loop b n body =
  if Int64.compare n 0L > 0 then
    ignore
      (B.for_ b ~lo:(B.index b 0L) ~hi:(B.index b n) ~init:B.Nil
         (fun b iv B.Nil ->
           body b iv;
           B.Nil))

let load ?(decode = Ssa_op.Decode.F32_to_f64) b buffer at =
  B.load_f64 b buffer ~decode (B.Flat at)

let store b buffer at x =
  B.store_f64 b buffer ~encode:Ssa_op.Encode.F32_round (B.Flat at) x

(* A cell rounded to binary32, then to bfloat16. *)
let store_bf16 b buffer at x =
  B.store_bf16 b buffer (B.Flat at) (B.f64_to_f32 b x)

let id (x : Ssa_buffer.t) = x.Ssa_buffer.id

(* The program [build] makes over [buffers]; a defect in the expansion, not a
   verdict about the input, when it does not verify. *)
let template ~buffers build =
  match Err.payload (B.program ~buffers build) with
  | Ok p -> p
  | Error e -> invalid_arg (Fmt.str "Ssa_expand: %a" Ssa_verify.pp_error e)

(* The declared buffers [ids] name, each once. *)
let declared (p : Ssa_program.t) ids =
  List.filter_map
    (Ssa_program.find_buffer p)
    (List.sort_uniq Ssa_id.Buffer.compare ids)

(* The template's body cloned where the operation stood, reading the effect the
   operation consumed; the effect it produced is redirected to the clone's last. *)
let splice t (template : Ssa_program.t) ~token ~after =
  let entry = template.Ssa_program.entry in
  let cl =
    Ssa_clone.create t ~subst:[ (List.hd entry.Ssa_region.params, token) ]
  in
  let stmts = Ssa_clone.stmts cl entry.Ssa_region.body in
  Ssa_rewrite.alias t ~from:after
    ~to_:(Ssa_clone.value cl (List.hd entry.Ssa_region.yields));
  stmts

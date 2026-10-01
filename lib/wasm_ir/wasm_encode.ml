let byte b n = Buffer.add_char b (Char.chr (n land 0xFF))

let uleb128 b (n : int64) =
  let rec go n =
    let low = Int64.to_int (Int64.logand n 0x7FL) in
    let rest = Int64.shift_right_logical n 7 in
    if Int64.equal rest 0L then byte b low
    else (
      byte b (low lor 0x80);
      go rest)
  in
  go n

let sleb128 b (n : int64) =
  let rec go n =
    let low = Int64.to_int (Int64.logand n 0x7FL) in
    let rest = Int64.shift_right n 7 in
    let sign = low land 0x40 <> 0 in
    if (Int64.equal rest 0L && not sign) || (Int64.equal rest (-1L) && sign)
    then byte b low
    else (
      byte b (low lor 0x80);
      go rest)
  in
  go n

(* Every index, count and length written here is non-negative and below 2^31:
   the validator bounds the ones a caller controls, and a buffer cannot hold
   more under a 32-bit [int]. *)
let u32 b n =
  if n < 0 then invalid_arg "Wasm_encode.u32: negative";
  uleb128 b (Int64.of_int n)

let name b s =
  u32 b (String.length s);
  Buffer.add_string b s

let vec b f l =
  u32 b (List.length l);
  List.iter (f b) l

let le32 b (x : int32) =
  for k = 0 to 3 do
    byte b (Int32.to_int (Int32.shift_right_logical x (8 * k)) land 0xFF)
  done

let le64 b (x : int64) =
  for k = 0 to 7 do
    byte b (Int64.to_int (Int64.shift_right_logical x (8 * k)) land 0xFF)
  done

let valtype b t = byte b (Wasm_type.to_byte t)

let block_type b : Wasm.Block_type.t -> unit = function
  | None -> byte b 0x40
  | Some t -> valtype b t

let mem_arg b (m : Wasm.Mem_arg.t) =
  u32 b m.Wasm.Mem_arg.align;
  u32 b m.Wasm.Mem_arg.offset

let rec instr b (i : Wasm.Instr.t) =
  match i with
  | Wasm.Instr.Block (bt, body) ->
      byte b 0x02;
      block_type b bt;
      List.iter (instr b) body;
      byte b 0x0B
  | Wasm.Instr.Br n ->
      byte b 0x0C;
      u32 b n
  | Wasm.Instr.Br_if n ->
      byte b 0x0D;
      u32 b n
  | Wasm.Instr.Call n ->
      byte b 0x10;
      u32 b n
  | Wasm.Instr.Drop -> byte b 0x1A
  | Wasm.Instr.F32_const x ->
      byte b 0x43;
      le32 b x
  | Wasm.Instr.F64_const x ->
      byte b 0x44;
      le64 b x
  | Wasm.Instr.Global_get n ->
      byte b 0x23;
      u32 b n
  | Wasm.Instr.Global_set n ->
      byte b 0x24;
      u32 b n
  | Wasm.Instr.I32_const x ->
      byte b 0x41;
      sleb128 b (Int64.of_int32 x)
  | Wasm.Instr.I64_const x ->
      byte b 0x42;
      sleb128 b x
  | Wasm.Instr.If (bt, yes, no) ->
      byte b 0x04;
      block_type b bt;
      List.iter (instr b) yes;
      if no <> [] then (
        byte b 0x05;
        List.iter (instr b) no);
      byte b 0x0B
  | Wasm.Instr.Load (l, m) ->
      byte b (Wasm.Load.byte l);
      mem_arg b m
  | Wasm.Instr.Local_get n ->
      byte b 0x20;
      u32 b n
  | Wasm.Instr.Local_set n ->
      byte b 0x21;
      u32 b n
  | Wasm.Instr.Local_tee n ->
      byte b 0x22;
      u32 b n
  | Wasm.Instr.Loop (bt, body) ->
      byte b 0x03;
      block_type b bt;
      List.iter (instr b) body;
      byte b 0x0B
  | Wasm.Instr.Memory_copy -> List.iter (byte b) [ 0xFC; 0x0A; 0x00; 0x00 ]
  | Wasm.Instr.Memory_fill -> List.iter (byte b) [ 0xFC; 0x0B; 0x00 ]
  | Wasm.Instr.Numeric op -> List.iter (byte b) (Wasm_op.bytes op)
  | Wasm.Instr.Return -> byte b 0x0F
  | Wasm.Instr.Select -> byte b 0x1B
  | Wasm.Instr.Store (s, m) ->
      byte b (Wasm.Store.byte s);
      mem_arg b m
  | Wasm.Instr.Unreachable -> byte b 0x00

let expr b (i : Wasm.Instr.t) =
  instr b i;
  byte b 0x0B

let func_type b (t : Wasm.Func_type.t) =
  byte b 0x60;
  vec b valtype t.Wasm.Func_type.params;
  vec b valtype t.Wasm.Func_type.results

(* Consecutive equal local types share one (count, type) run. *)
let local_runs ts =
  List.fold_left
    (fun acc t ->
      match acc with
      | (n, u) :: rest when Wasm_type.equal t u -> (n + 1, u) :: rest
      | _ -> (1, t) :: acc)
    [] ts
  |> List.rev

let section out id f =
  let b = Buffer.create 64 in
  f b;
  byte out id;
  u32 out (Buffer.length b);
  Buffer.add_buffer out b

let index_of equal x l =
  let rec go i = function
    | [] -> invalid_arg "Wasm_encode: function type not interned"
    | y :: rest -> if equal x y then i else go (i + 1) rest
  in
  go 0 l

let intern_types (m : Wasm.Module.t) =
  let all =
    List.map (fun i -> i.Wasm.Import.type_) m.Wasm.Module.imports
    @ List.map (fun f -> f.Wasm.Func.type_) m.Wasm.Module.funcs
  in
  List.fold_left
    (fun acc t ->
      if List.exists (Wasm.Func_type.equal t) acc then acc else acc @ [ t ])
    [] all

let write (m : Wasm.Module.t) =
  let out = Buffer.create 256 in
  Buffer.add_string out "\000asm\001\000\000\000";
  let types = intern_types m in
  let type_index t = index_of Wasm.Func_type.equal t types in
  if types <> [] then section out 1 (fun b -> vec b func_type types);
  if m.Wasm.Module.imports <> [] then
    section out 2 (fun b ->
        vec b
          (fun b (i : Wasm.Import.t) ->
            name b i.Wasm.Import.module_name;
            name b i.Wasm.Import.name;
            byte b 0x00;
            u32 b (type_index i.Wasm.Import.type_))
          m.Wasm.Module.imports);
  if m.Wasm.Module.funcs <> [] then
    section out 3 (fun b ->
        vec b
          (fun b (f : Wasm.Func.t) -> u32 b (type_index f.Wasm.Func.type_))
          m.Wasm.Module.funcs);
  (match m.Wasm.Module.memory with
  | None -> ()
  | Some { Wasm.Memory.min_pages; max_pages } ->
      section out 5 (fun b ->
          u32 b 1;
          match max_pages with
          | None ->
              byte b 0x00;
              u32 b min_pages
          | Some x ->
              byte b 0x01;
              u32 b min_pages;
              u32 b x));
  if m.Wasm.Module.globals <> [] then
    section out 6 (fun b ->
        vec b
          (fun b (g : Wasm.Global.t) ->
            valtype b g.Wasm.Global.type_;
            byte b (if g.Wasm.Global.mutable_ then 1 else 0);
            expr b g.Wasm.Global.init)
          m.Wasm.Module.globals);
  if m.Wasm.Module.exports <> [] then
    section out 7 (fun b ->
        vec b
          (fun b (e : Wasm.Export.t) ->
            name b e.Wasm.Export.name;
            match e.Wasm.Export.kind with
            | Wasm.Export.Func n ->
                byte b 0x00;
                u32 b n
            | Wasm.Export.Global n ->
                byte b 0x03;
                u32 b n
            | Wasm.Export.Memory ->
                byte b 0x02;
                u32 b 0)
          m.Wasm.Module.exports);
  if m.Wasm.Module.funcs <> [] then
    section out 10 (fun b ->
        vec b
          (fun b (f : Wasm.Func.t) ->
            let body = Buffer.create 64 in
            vec body
              (fun b (n, t) ->
                u32 b n;
                valtype b t)
              (local_runs f.Wasm.Func.locals);
            List.iter (instr body) f.Wasm.Func.body;
            byte body 0x0B;
            u32 b (Buffer.length body);
            Buffer.add_buffer b body)
          m.Wasm.Module.funcs);
  if m.Wasm.Module.data <> [] then
    section out 11 (fun b ->
        vec b
          (fun b (d : Wasm.Data.t) ->
            byte b 0x00;
            expr b (Wasm.Instr.I32_const (Int32.of_int d.Wasm.Data.offset));
            u32 b (String.length d.Wasm.Data.bytes);
            Buffer.add_string b d.Wasm.Data.bytes)
          m.Wasm.Module.data);
  List.iter
    (fun (c : Wasm.Custom.t) ->
      section out 0 (fun b ->
          name b c.Wasm.Custom.name;
          Buffer.add_string b c.Wasm.Custom.payload))
    m.Wasm.Module.customs;
  Buffer.contents out

let module_ m = Err.map (fun () -> write m) (Wasm_check.module_ m)

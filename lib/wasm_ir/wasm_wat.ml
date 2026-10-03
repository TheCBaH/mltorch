let pp_types = Fmt.(list ~sep:(any " ") Wasm_type.pp)

let pp_func_type ppf (t : Wasm.Func_type.t) =
  let group name ppf = function
    | [] -> ()
    | ts -> Fmt.pf ppf " (%s %a)" name pp_types ts
  in
  Fmt.pf ppf "%a%a" (group "param") t.Wasm.Func_type.params (group "result")
    t.Wasm.Func_type.results

let pp_bt ppf : Wasm.Block_type.t -> unit = function
  | None -> ()
  | Some t -> Fmt.pf ppf " (result %a)" Wasm_type.pp t

let pp_arg ppf (m : Wasm.Mem_arg.t) =
  Fmt.pf ppf " offset=%d align=%d" m.Wasm.Mem_arg.offset
    (1 lsl m.Wasm.Mem_arg.align)

let rec pp_instr ~depth ppf (i : Wasm.Instr.t) =
  let pad = String.make (2 * depth) ' ' in
  let line fmt = Fmt.pf ppf ("%s" ^^ fmt ^^ "@,") pad in
  let body l = List.iter (pp_instr ~depth:(depth + 1) ppf) l in
  match i with
  | Wasm.Instr.Block (bt, l) ->
      line "block%a" pp_bt bt;
      body l;
      line "end"
  | Wasm.Instr.Br n -> line "br %d" n
  | Wasm.Instr.Br_if n -> line "br_if %d" n
  | Wasm.Instr.Call n -> line "call %d" n
  | Wasm.Instr.Drop -> line "drop"
  | Wasm.Instr.F32_const x -> line "f32.const 0x%08lx" x
  | Wasm.Instr.F64_const x ->
      line "f64.const 0x%016Lx (%h)" x (Int64.float_of_bits x)
  | Wasm.Instr.Global_get n -> line "global.get %d" n
  | Wasm.Instr.Global_set n -> line "global.set %d" n
  | Wasm.Instr.I32_const x -> line "i32.const %ld" x
  | Wasm.Instr.I64_const x -> line "i64.const %Ld" x
  | Wasm.Instr.If (bt, yes, no) ->
      line "if%a" pp_bt bt;
      body yes;
      if no <> [] then (
        line "else";
        body no);
      line "end"
  | Wasm.Instr.Load (l, m) -> line "%s%a" (Wasm.Load.name l) pp_arg m
  | Wasm.Instr.Local_get n -> line "local.get %d" n
  | Wasm.Instr.Local_set n -> line "local.set %d" n
  | Wasm.Instr.Local_tee n -> line "local.tee %d" n
  | Wasm.Instr.Loop (bt, l) ->
      line "loop%a" pp_bt bt;
      body l;
      line "end"
  | Wasm.Instr.Memory_copy -> line "memory.copy"
  | Wasm.Instr.Memory_fill -> line "memory.fill"
  | Wasm.Instr.Numeric op -> line "%s" (Wasm_op.name op)
  | Wasm.Instr.Return -> line "return"
  | Wasm.Instr.Select -> line "select"
  | Wasm.Instr.Simd_lane (op, lane) ->
      line "%s %d" (Wasm.Simd_lane.name op) lane
  | Wasm.Instr.Simd_load (l, m) -> line "%s%a" (Wasm.Simd_load.name l) pp_arg m
  | Wasm.Instr.Simd_store (s, m, lane) ->
      if Wasm.Simd_store.lanes s > 0 then
        line "%s%a %d" (Wasm.Simd_store.name s) pp_arg m lane
      else line "%s%a" (Wasm.Simd_store.name s) pp_arg m
  | Wasm.Instr.V128_const bytes ->
      line "v128.const i8x16 %s"
        (String.concat " "
           (List.init (String.length bytes) (fun i ->
                string_of_int (Char.code bytes.[i]))))
  | Wasm.Instr.Store (s, m) -> line "%s%a" (Wasm.Store.name s) pp_arg m
  | Wasm.Instr.Unreachable -> line "unreachable"

let pp ppf (m : Wasm.Module.t) =
  Fmt.pf ppf "@[<v>(module@,";
  List.iteri
    (fun k (i : Wasm.Import.t) ->
      Fmt.pf ppf "  (import %S %S (func %d%a))@," i.Wasm.Import.module_name
        i.Wasm.Import.name k pp_func_type i.Wasm.Import.type_)
    m.Wasm.Module.imports;
  (match m.Wasm.Module.memory with
  | None -> ()
  | Some { Wasm.Memory.min_pages; max_pages } ->
      Fmt.pf ppf "  (memory %d%a)@," min_pages
        Fmt.(option (any " " ++ int))
        max_pages);
  List.iteri
    (fun k (g : Wasm.Global.t) ->
      Fmt.pf ppf "  (global %d %s%a)@," k
        (if g.Wasm.Global.mutable_ then "mut " else "")
        Wasm_type.pp g.Wasm.Global.type_)
    m.Wasm.Module.globals;
  let base = List.length m.Wasm.Module.imports in
  List.iteri
    (fun k (f : Wasm.Func.t) ->
      Fmt.pf ppf "  (func %d%a@," (base + k) pp_func_type f.Wasm.Func.type_;
      if f.Wasm.Func.locals <> [] then
        Fmt.pf ppf "    (local %a)@," pp_types f.Wasm.Func.locals;
      List.iter (pp_instr ~depth:2 ppf) f.Wasm.Func.body;
      Fmt.pf ppf "  )@,")
    m.Wasm.Module.funcs;
  List.iter
    (fun (e : Wasm.Export.t) ->
      let kind, n =
        match e.Wasm.Export.kind with
        | Wasm.Export.Func n -> ("func", n)
        | Wasm.Export.Global n -> ("global", n)
        | Wasm.Export.Memory -> ("memory", 0)
      in
      Fmt.pf ppf "  (export %S (%s %d))@," e.Wasm.Export.name kind n)
    m.Wasm.Module.exports;
  List.iter
    (fun (d : Wasm.Data.t) ->
      Fmt.pf ppf "  (data (offset %d) %d bytes)@," d.Wasm.Data.offset
        (String.length d.Wasm.Data.bytes))
    m.Wasm.Module.data;
  Fmt.pf ppf ")@]"

let to_string m = Fmt.str "%a" pp m

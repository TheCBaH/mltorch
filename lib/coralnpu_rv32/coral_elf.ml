open Err.Syntax

let abi =
  {
    Elf.Riscv.xlen = Rv32;
    float_abi = Single;
    rvc = false;
    rve = false;
    tso = false;
    arch =
      Elf.Riscv.isa_string Rv32
        ~base:(`I, (2, 1))
        [
          ("m", (2, 0)); ("f", (2, 2)); ("zicsr", (2, 0)); ("zifencei", (2, 0));
        ];
    stack_align = 16;
    unaligned_access = None;
  }

module Unloadable = struct
  type t = { address : int64; name : string; size : int64 }
end

type error =
  [ `Elf of Elf.Exec.error | `No_entry | `Unloadable_segment of Unloadable.t ]

let pp_error ppf : error -> unit = function
  | `Elf e -> Elf.Exec.pp_error ppf e
  | `No_entry -> Fmt.string ppf "the image has no entry"
  | `Unloadable_segment { Unloadable.address; name; size } ->
      Fmt.pf ppf
        "segment %s (%Ld bytes at 0x%08Lx) is outside every loadable region"
        name size address

let within (r : Coral_profile.Region.t) ~address ~size =
  address >= r.base && Int64.add address size <= Int64.add r.base r.size

(* The loadable region holding [size] bytes at [address]. *)
let home (p : Coral_profile.t) ~name ~address ~size =
  List.find_opt
    (fun (r : Coral_profile.Region.t) -> r.loadable && within r ~address ~size)
    p.regions
  |> Err.of_option (`Unloadable_segment { Unloadable.address; name; size })

let htif_names = [ "tohost"; "fromhost"; "tohost_ready"; "fromhost_ready" ]

(* The RTL simulators fill unloaded TCM with different bytes on every run, and
   the core fetches up to 76 bytes past the last instruction: a stray word
   there that predecodes as a jump costs a mispredicted fetch, 3 cycles. So
   every PT_LOAD spans [fill_align]-aligned bounds, with at least
   [fill_margin] bytes of zeros past its last section, within its region. *)
let fill_align = 128L
let fill_margin = 128L
let align_down a = Int64.logand a (Int64.neg fill_align)
let align_up a = align_down (Int64.add a (Int64.pred fill_align))

let section_size (s : Elf.Exec.section) =
  match s.content with Bytes b -> Int64.of_int (String.length b) | Zero n -> n

(* [sections], in address order, between the zero sections that fill them out
   to [r]'s aligned bounds. *)
let fill (r : Coral_profile.Region.t) sections =
  let zeros name addr size =
    if size <= 0L then []
    else
      [
        {
          Elf.Exec.name = name ^ r.name;
          addr;
          align = 1L;
          content = Bytes (String.make (Int64.to_int size) '\000');
        };
      ]
  in
  match (sections, List.rev sections) with
  | (first : Elf.Exec.section) :: _, (last : Elf.Exec.section) :: _ ->
      let lo = max r.base (align_down first.addr) in
      let stop = Int64.add last.addr (section_size last) in
      let hi =
        min (Int64.add r.base r.size) (align_up (Int64.add stop fill_margin))
      in
      zeros ".fill_head." lo (Int64.sub first.addr lo)
      @ sections
      @ zeros ".fill." stop (Int64.sub hi stop)
  | _ -> sections

let write (p : Coral_profile.t) (l : Coral_layout.t) (image : Image.t) =
  let* entry = Err.of_option `No_entry image.entry in
  let image_sections =
    List.filter_map
      (fun (s : Image.segment) ->
        if String.length s.bytes = 0 && s.zero_fill = 0 then None
        else
          Some
            ( s.perms,
              {
                Elf.Exec.name = s.name;
                addr = s.address;
                align = Int64.of_int (max 1 s.alignment);
                content = Bytes (s.bytes ^ String.make s.zero_fill '\000');
              } ))
      image.segments
  in
  let htif =
    ( Asm_core.Perms.rw,
      {
        Elf.Exec.name = ".htif";
        addr = l.htif;
        align = 64L;
        content =
          Bytes (String.make (Int64.to_int Coral_layout.htif_size) '\000');
      } )
  in
  let* placed =
    Err.List.map
      (fun ((perms : Asm_core.Perms.t), (s : Elf.Exec.section)) ->
        let* r = home p ~name:s.name ~address:s.addr ~size:(section_size s) in
        Err.return (r.Coral_profile.Region.name, perms, s))
      (image_sections @ [ htif ])
  in
  let segments =
    List.filter_map
      (fun (r : Coral_profile.Region.t) ->
        match List.filter (fun (n, _, _) -> n = r.name) placed with
        | [] -> None
        | mine ->
            let any f = List.exists (fun (_, perms, _) -> f perms) mine in
            Some
              {
                Elf.Exec.perms =
                  {
                    read = any (fun (q : Asm_core.Perms.t) -> q.read);
                    write = any (fun (q : Asm_core.Perms.t) -> q.write);
                    execute = any (fun (q : Asm_core.Perms.t) -> q.execute);
                  };
                paddr = None;
                sections =
                  fill r
                    (List.sort
                       (fun (a : Elf.Exec.section) b ->
                         Int64.compare a.addr b.addr)
                       (List.map (fun (_, _, s) -> s) mine));
              })
      p.regions
  in
  let entry_section =
    List.find_map
      (fun (_, _, (s : Elf.Exec.section)) ->
        match s.content with
        | Bytes b
          when entry >= s.addr
               && entry < Int64.add s.addr (Int64.of_int (String.length b)) ->
            Some s.name
        | Bytes _ | Zero _ -> None)
      placed
  in
  let symbol ?(kind = `Object) ?(size = 8L) ~section name value =
    { Elf.Exec.name; value; size; kind; binding = `Global; section }
  in
  let symbols =
    List.mapi
      (fun i name ->
        symbol ~section:(Some ".htif") name
          (Int64.add l.htif (Int64.of_int (64 * i))))
      htif_names
    @ [ symbol ~kind:`Func ~size:0L ~section:entry_section "_start" entry ]
  in
  Elf.Exec.build
    {
      Elf.Exec.target = Elf.Target.riscv abi;
      entry;
      segments;
      headers = Unmapped;
      gnu_stack = None;
      section_headers = true;
      symbols;
    }
  |> Err.map_error ~pos:__POS__ (fun e -> `Elf e)

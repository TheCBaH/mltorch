(* [model_main.c]: command line, payload validation and mapping, workspace
   allocation, output publication and cleanup. It knows nothing of the model but
   what [C_model_abi] declares; the tensors' offsets stay inside [model_run].
   Exit codes are [C_model_abi.exit_*]. *)

let body =
  {|
static const char *g_prog = "model";

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static void die_usage(void) {
  fprintf(stderr,
          "usage: %s --weights FILE --inputs FILE --outputs FILE [--poison] [--time] "
          "[--repeat N]\n",
          g_prog);
}

/* Map a whole payload file read-only and check it is the one this binary was
   generated for. Returns 0, or -1 after a message. */
static int map_payload(const char *what, const char *path, uint64_t expect,
                       const unsigned char *header, const void **out,
                       size_t *len) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "%s: cannot open %s file %s: %s\n", g_prog, what, path,
            strerror(errno));
    return -1;
  }
  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    fprintf(stderr, "%s: %s file %s is not a regular file\n", g_prog, what,
            path);
    close(fd);
    return -1;
  }
  if (st.st_size < 64) {
    fprintf(stderr, "%s: %s file %s is truncated (%lld bytes)\n", g_prog, what,
            path, (long long)st.st_size);
    close(fd);
    return -1;
  }
  if ((uint64_t)st.st_size != expect) {
    fprintf(stderr, "%s: %s file %s has %llu bytes, expected %llu\n", g_prog,
            what, path, (unsigned long long)st.st_size,
            (unsigned long long)expect);
    close(fd);
    return -1;
  }
  void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) {
    fprintf(stderr, "%s: cannot map %s file %s: %s\n", g_prog, what, path,
            strerror(errno));
    return -1;
  }
  const unsigned char *h = (const unsigned char *)p;
  if (memcmp(h, header, 8) != 0) {
    fprintf(stderr, "%s: %s file %s is not a payload file\n", g_prog, what,
            path);
  } else if (memcmp(h + 12, header + 12, 4) != 0) {
    fprintf(stderr, "%s: %s file %s has the wrong payload role\n", g_prog,
            what, path);
  } else if (memcmp(h + 32, header + 32, 16) != 0) {
    fprintf(stderr, "%s: %s file %s was made for a different model\n", g_prog,
            what, path);
  } else if (memcmp(h, header, 64) != 0) {
    fprintf(stderr, "%s: %s file %s has an incompatible header\n", g_prog, what,
            path);
  } else {
    *out = p;
    *len = (size_t)st.st_size;
    return 0;
  }
  munmap(p, (size_t)st.st_size);
  return -1;
}

static int same_file(const char *a, const char *b) {
  struct stat sa, sb;
  if (stat(a, &sa) != 0 || stat(b, &sb) != 0) return 0;
  return sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;
}

int main(int argc, char **argv) {
  g_prog = argc > 0 ? argv[0] : "model";
  const char *weights_path = NULL, *inputs_path = NULL, *outputs_path = NULL;
  int poison = 0, timing = 0, repeat = 0;
  const double t_start = now_ms();
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--poison") == 0) {
      poison = 1;
    } else if (strcmp(argv[i], "--time") == 0) {
      timing = 1;
    } else if (i + 1 < argc && strcmp(argv[i], "--repeat") == 0) {
      repeat = atoi(argv[++i]);
      if (repeat < 0) repeat = 0;
      timing = 1;
    } else if (i + 1 < argc && strcmp(argv[i], "--weights") == 0) {
      weights_path = argv[++i];
    } else if (i + 1 < argc && strcmp(argv[i], "--inputs") == 0) {
      inputs_path = argv[++i];
    } else if (i + 1 < argc && strcmp(argv[i], "--outputs") == 0) {
      outputs_path = argv[++i];
    } else {
      die_usage();
      return EXIT_USAGE;
    }
  }
  if (!weights_path || !inputs_path || !outputs_path) {
    die_usage();
    return EXIT_USAGE;
  }
  if (same_file(outputs_path, weights_path) ||
      same_file(outputs_path, inputs_path)) {
    fprintf(stderr, "%s: the output file aliases an input file\n", g_prog);
    return EXIT_PAYLOAD;
  }

  int rc = EXIT_PAYLOAD;
  const void *weights = NULL, *inputs = NULL;
  size_t weights_len = 0, inputs_len = 0;
  void *workspace = NULL, *outputs = NULL;
  char *tmp_path = NULL;
  int out_fd = -1;
  size_t out_len = (size_t)model_outputs_size;

  if (map_payload("weights", weights_path, model_weights_size,
                  model_weights_header, &weights, &weights_len) != 0)
    goto done;
  if (map_payload("inputs", inputs_path, model_inputs_size,
                  model_inputs_header, &inputs, &inputs_len) != 0)
    goto done;

  /* Outputs are built in a temporary sibling and renamed on success only. A
     stale file at the destination is removed first: after any failure nothing
     is left there for a caller to mistake for a result. */
  unlink(outputs_path);
  size_t n = strlen(outputs_path) + 32;
  tmp_path = malloc(n);
  if (!tmp_path) {
    rc = EXIT_ALLOCATION;
    goto done;
  }
  snprintf(tmp_path, n, "%s.tmp.%ld", outputs_path, (long)getpid());
  out_fd = open(tmp_path, O_RDWR | O_CREAT | O_EXCL, 0644);
  if (out_fd < 0 || ftruncate(out_fd, (off_t)out_len) != 0) {
    fprintf(stderr, "%s: cannot create output file %s: %s\n", g_prog,
            tmp_path, strerror(errno));
    goto done;
  }
  outputs = mmap(NULL, out_len, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);
  if (outputs == MAP_FAILED) {
    outputs = NULL;
    fprintf(stderr, "%s: cannot map output file: %s\n", g_prog,
            strerror(errno));
    goto done;
  }
  memcpy(outputs, model_outputs_header, 64);

  {
    size_t align = (size_t)model_workspace_alignment;
    if (align < sizeof(void *)) align = sizeof(void *);
    size_t size = (size_t)model_workspace_size;
    if (posix_memalign(&workspace, align, size ? size : 1) != 0) {
      workspace = NULL;
      fprintf(stderr, "%s: cannot allocate a %llu-byte workspace\n", g_prog,
              (unsigned long long)model_workspace_size);
      rc = EXIT_ALLOCATION;
      goto done;
    }
    if (poison) memset(workspace, 0xA5, size);
  }

  {
    struct model_error err;
    memset(&err, 0, sizeof err);
    const double t_run = now_ms();
    if (model_run(weights, inputs, workspace, outputs, &err) != 0) {
      fprintf(stderr, "%s: inference failed in invocation %d (kind %d)\n",
              g_prog, (int)err.invocation, (int)err.kind);
      fprintf(stderr, "model_error: %d %d", (int)err.invocation, (int)err.kind);
      for (int i = 0; i < MODEL_ERROR_WORDS; i++)
        fprintf(stderr, " %lld", (long long)err.v[i]);
      fprintf(stderr, "\n");
      rc = EXIT_INFERENCE;
      goto done;
    }
    if (timing) {
      const double t_first = now_ms();
      /* Warm repeats over the same (now dirty) workspace, on the same inputs:
         the timing of the inference unit alone, without process start-up,
         mapping or output publication. */
      double best = -1.0, total = 0.0;
      for (int r = 0; r < repeat; r++) {
        const double a = now_ms();
        if (model_run(weights, inputs, workspace, outputs, &err) != 0) {
          rc = EXIT_INFERENCE;
          goto done;
        }
        const double d = now_ms() - a;
        total += d;
        if (best < 0.0 || d < best) best = d;
      }
      fprintf(stderr,
              "model_time_ms: setup %.3f first_run %.3f warm_best %.3f "
              "warm_mean %.3f repeats %d\n",
              t_run - t_start, t_first - t_run, best < 0.0 ? 0.0 : best,
              repeat ? total / repeat : 0.0, repeat);
    }
  }

  if (msync(outputs, out_len, MS_SYNC) != 0 ||
      munmap(outputs, out_len) != 0) {
    outputs = NULL;
    fprintf(stderr, "%s: cannot flush the output file: %s\n", g_prog,
            strerror(errno));
    goto done;
  }
  outputs = NULL;
  if (close(out_fd) != 0) {
    out_fd = -1;
    fprintf(stderr, "%s: cannot close the output file: %s\n", g_prog,
            strerror(errno));
    goto done;
  }
  out_fd = -1;
  if (rename(tmp_path, outputs_path) != 0) {
    fprintf(stderr, "%s: cannot publish %s: %s\n", g_prog, outputs_path,
            strerror(errno));
    goto done;
  }
  free(tmp_path);
  tmp_path = NULL;
  rc = EXIT_OK;

done:
  if (outputs) munmap(outputs, out_len);
  if (out_fd >= 0) close(out_fd);
  if (tmp_path) {
    unlink(tmp_path);
    free(tmp_path);
  }
  free(workspace);
  if (inputs) munmap((void *)inputs, inputs_len);
  if (weights) munmap((void *)weights, weights_len);
  return rc;
}
|}

let source =
  String.concat "\n"
    [
      "#define _POSIX_C_SOURCE 200809L";
      "#include <errno.h>";
      "#include <fcntl.h>";
      "#include <stdio.h>";
      "#include <stdlib.h>";
      "#include <sys/mman.h>";
      "#include <sys/stat.h>";
      "#include <time.h>";
      "#include <unistd.h>";
      Loop_c_runtime.prelude;
      C_model_abi.declarations;
      Printf.sprintf "#define EXIT_OK %d" C_model_abi.exit_ok;
      Printf.sprintf "#define EXIT_USAGE %d" C_model_abi.exit_usage;
      Printf.sprintf "#define EXIT_PAYLOAD %d" C_model_abi.exit_payload;
      Printf.sprintf "#define EXIT_ALLOCATION %d" C_model_abi.exit_allocation;
      Printf.sprintf "#define EXIT_INFERENCE %d" C_model_abi.exit_inference;
      body;
    ]

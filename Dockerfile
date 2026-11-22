# Shopping Sessions: Rust pipeline, built with profile-guided optimization (PGO).
# Training data comes from the organizers' published generator (generator/), made
# at build time; nothing is checked in. If any PGO step fails, the image falls
# back to a normal optimized build.
FROM rust:bookworm AS build
WORKDIR /src
ENV BASEFLAGS="-C target-cpu=x86-64-v3"

# 1. generator for training data
COPY generator /gen
RUN cd /gen && cargo build --release --locked && cp target/release/generate /usr/local/bin/generate

# 2. sources
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# 3. PGO: instrumented build -> training run -> merged profile (non-fatal)
RUN set -e; \
    ( rustup component add llvm-tools \
      && RUSTFLAGS="$BASEFLAGS -Cprofile-generate=/tmp/pgo" cargo build --release --locked --target-dir /tmp/inst \
      && generate --seed 5 --events 2000000 --clicks 500000 --output /tmp/train \
      && MUTATE_HEX=/tmp/train /tmp/inst/release/sess \
      && rm -f /tmp/pgo/*.profraw \
      && mkdir -p /tmp/out \
      && DATA_DIR=/tmp/train OUTPUT_DIR=/tmp/out NCPU=4 /tmp/inst/release/sess \
      && PROFDATA=$(find "$(rustc --print sysroot)" -name llvm-profdata -type f | head -1) \
      && "$PROFDATA" merge -o /tmp/pgo.profdata /tmp/pgo/*.profraw \
      && echo "PGO profile ready" ) || echo "PGO failed, building without it"; \
    rm -rf /tmp/inst /tmp/train /tmp/out

# 4. final build (with the profile if it exists). On-scorer measurements showed the AVX2
#    build slightly faster than an AVX-512 (Cascade Lake) build, so only AVX2 is shipped.
RUN PGO=""; if [ -f /tmp/pgo.profdata ]; then PGO="-Cprofile-use=/tmp/pgo.profdata"; fi; \
    ( RUSTFLAGS="$BASEFLAGS $PGO" cargo build --release --locked \
      || RUSTFLAGS="$BASEFLAGS" cargo build --release --locked ) \
    && cp target/release/sess /src/sess_v3

FROM debian:bookworm-slim
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
COPY --from=build /src/sess_v3 /app/sess
USER 1000
ENV MIMALLOC_ALLOW_LARGE_OS_PAGES=1
CMD ["/app/sess"]

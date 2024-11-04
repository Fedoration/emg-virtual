from pylsl import StreamInlet, resolve_streams

print('Search started')
streams = resolve_streams()
for stream in streams:
    print(stream.name())
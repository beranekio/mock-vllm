package text

// Chunk splits s into chunks of up to size runes (not bytes).
func Chunk(s string, size int) []string {
	if s == "" {
		return nil
	}
	if size < 1 {
		size = 1
	}
	runes := []rune(s)
	var chunks []string
	for i := 0; i < len(runes); i += size {
		end := i + size
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
	}
	return chunks
}
